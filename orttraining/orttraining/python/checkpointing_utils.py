import os
from collections import OrderedDict
import torch

def list_checkpoint_files(checkpoint_dir, checkpoint_prefix, extension='.ort.pt'):
    ckpt_file_names = [f for f in os.listdir(checkpoint_dir) if f.startswith(checkpoint_prefix)]
    ckpt_file_names = [f for f in ckpt_file_names if f.endswith(extension)]
    ckpt_file_names = [os.path.join(checkpoint_dir, f) for f in ckpt_file_names]
    
    assert len(ckpt_file_names) > 0, "No checkpoint files found with prefix \"{}\" in directory {}.".format(checkpoint_prefix, checkpoint_dir)
    return ckpt_file_names

def get_checkpoint_name(prefix, zero_enabled, world_rank=0, world_size=1, horizontal_parallel_size=1, pipeline_parallel_size=1):
    data_parallel_size = world_size / horizontal_parallel_size / pipeline_parallel_size
    parallellism_info = 'D.{data_parallel_size}.H.{horizontal_parallel_size}.P.{pipeline_parallel_size}'
    SINGLE_CHECKPOINT_FILENAME='{prefix}.ort.pt'
    MULTIPLE_CHECKPOINT_FILENAME='{prefix}.rank.{world_rank}.{world_size}.' + parallellism_info + '.ort.pt'
    
    is_partitioned = zero_enabled or (horizontal_parallel_size > 1) or (pipeline_parallel_size > 1)
    # if is_partitioned:
    #     filename=MULTIPLE_CHECKPOINT_FILENAME.format(prefix=prefix, world_rank=world_rank, world_size=(world_size-1),
    #         data_parallel_size=data_parallel_size, 
    #         horizontal_parallel_size=horizontal_parallel_size, 
    #         pipeline_parallel_size=pipeline_parallel_size)
    # else:
    #     filename=SINGLE_CHECKPOINT_FILENAME.format(prefix=prefix)
    filename=MULTIPLE_CHECKPOINT_FILENAME.format(prefix=prefix, world_rank=world_rank, world_size=(world_size-1),
        data_parallel_size=data_parallel_size, 
        horizontal_parallel_size=horizontal_parallel_size, 
        pipeline_parallel_size=pipeline_parallel_size)

    return filename

def split_state_dict(state_dict):
    optimizer_keys = ['Moment_', 'Update_Count_', 'Step_']
    split_sd = {'optimizer': {}, 'fp32_param': {}, 'fp16_param': {}}
    for k,v in state_dict.items():
        mode = 'fp32_param'
        for optim_key in optimizer_keys:
            if k.startswith(optim_key):
                mode = 'optimizer'
                break
        if k.endswith('_fp16'):
            mode = 'fp16_param'
        split_sd[mode][k] = v
    
    return split_sd

def is_equal(A, B):
    try:
        assert A.keys() == B.keys()
        for k in A.keys():
            assert (A[k] == B[k]).all()
    except:
        return False
    return True

def is_equal_tensor(A, B):
    return (A == B).all()

class CombineZeroCheckpoint(object):
    def __init__(self, checkpoint_files, clean_state_dict = None):

        assert len(checkpoint_files) > 0, "No checkpoint files passed."

        self.checkpoint_files = checkpoint_files
        self.clean_state_dict = clean_state_dict
        self.world_size = int(self.checkpoint_files[0].split('rank')[1].split('.')[2]) +1
        print(f"World size = {self.world_size}, expecting {self.world_size} files.")        
        assert len(self.checkpoint_files) == self.world_size, "Could not find {} files".format(self.world_size)
        
        self.weight_shape_map = dict()
    
    def _is_sharded(self, name):
        if '_view_' in name:
            return True
        return False

    def _has_fp16_weights(self, state_dict):
        for k in state_dict.keys():
            if k.endswith('_fp16'):
                return True
        return False
    
    def _split_moment_name(self, name):
        name_split = name.split('_view_')
        if(len(name_split)>1):
            view_num = int(name_split[1])
        else:
            view_num = None
        weight_name = name_split[0].split('Moment_')[1][2:]
        moment_num = int(name_split[0].split('Moment_')[1][0])
        return moment_num, weight_name, view_num

    def _update_weight_statistics(self, name, value):
        self.weight_shape_map[name] = value.size() #original shape of tensor

    def _reshape_tensors(self, state_dict, fp16):
        for k,v in state_dict.items():
            if k.startswith('Moment_'):
                _, weight_name, _ = self._split_moment_name(k)
                set_size = self.weight_shape_map[weight_name]    
                state_dict[k] = v.reshape(set_size)
                state_dict[weight_name] = state_dict[weight_name].reshape(set_size)
        return state_dict
  
    def aggregate_checkpoints(self):
        checkpoint_dir=os.path.dirname(self.checkpoint_files[0])
        checkpoint_prefix = self.checkpoint_files[0].split('.rank')[0]
        self.aggregate_state_dict=dict()

        is_fp16 = False
        weight_offset = dict()
        for i in range(self.world_size):
            checkpoint_name = get_checkpoint_name(checkpoint_prefix, True, i, self.world_size)
            print("Loading state dict from: {}".format(checkpoint_name))
            rank_state_dict = torch.load(checkpoint_name, map_location=torch.device("cpu"))
            if 'model' in rank_state_dict:
                rank_state_dict = rank_state_dict['model']
            
            if self.clean_state_dict:
                rank_state_dict = self.clean_state_dict(rank_state_dict)
            
            if i==0:
                is_fp16 = self._has_fp16_weights(rank_state_dict)

            for k,v in rank_state_dict.items():
                if k.startswith('Moment_'):
                    moment_num, weight_name, view_num = self._split_moment_name(k)

                    if self._is_sharded(k):                                                
                        clean_name = 'Moment_' + str(moment_num) + '_' + weight_name
                        if clean_name in self.aggregate_state_dict:
                            # Found a previous shard of the moment, concatenate shards ordered by ranks
                            self.aggregate_state_dict[clean_name] = torch.cat((self.aggregate_state_dict[clean_name], v), 0)
                        else:
                            self.aggregate_state_dict[clean_name] = v
                    else:
                        # Moment is not sharded, add as is                        
                        self.aggregate_state_dict[k] = v

                    if is_fp16 and moment_num == 1:
                        #FP32 weights are sharded, patch together based on moments
                        if view_num == 0:
                            # This FP32 weight's first shard is present on this rank, 
                            # flatten and add the weight's first view
                            self.aggregate_state_dict[weight_name] = rank_state_dict[weight_name].view(-1)                                       
                            self._update_weight_statistics(weight_name, rank_state_dict[weight_name])
                            weight_offset[weight_name] = v.numel()
                        
                        elif view_num == 1:
                            # This FP32 weight is carryforward from previous rank
                            # Get start and end of weight slice to be updated from this rank
                            weight_start = weight_offset[weight_name]
                            weight_end = weight_start + v.numel()

                            if weight_start:
                                old_value = self.aggregate_state_dict[weight_name]
                                new_value = rank_state_dict[weight_name].view(-1)                             
                                # patch the weight together
                                self.aggregate_state_dict[weight_name] = torch.cat((old_value[:weight_start], new_value[weight_start:weight_end], old_value[weight_end:]),0)
                            
                            # update offset for next view
                            weight_offset[weight_name] = weight_end

                elif k.startswith('Update_Count'):
                    clean_name = k.split('_view_')[0]
                    # add a single copy of the 'Update_Count' tensor for current weight                    
                    if clean_name not in self.aggregate_state_dict:
                        self.aggregate_state_dict[clean_name] = v
                
                else:                    
                    if k not in self.aggregate_state_dict:
                        self.aggregate_state_dict[k] = v 
                        if not (k.endswith('_fp16') or k == 'Step'):
                            # FP32 Weight
                            self._update_weight_statistics(k,v)

        final_state_dict = self._reshape_tensors(self.aggregate_state_dict, is_fp16)
        return final_state_dict

class CombineMegatronCheckpoint(object):
    def __init__(self, checkpoint_files, clean_state_dict = None):

        assert len(checkpoint_files) > 0, "No checkpoint files passed."

        self.checkpoint_files = checkpoint_files
        self.clean_state_dict = clean_state_dict
        filename = os.path.basename(self.checkpoint_files[0])
        self.checkpoint_prefix = self.checkpoint_files[0].split('.rank')[0]
        self.world_size = int(filename.split('rank')[1].split('.')[2]) +1
        self.D_size = int(filename.split('.D.')[1].split('.')[0])
        self.H_size = int(filename.split('.H.')[1].split('.')[0])
        self.P_size = int(filename.split('.P.')[1].split('.')[0])
        # print(f"World size = {self.world_size}.")        
        # assert len(self.checkpoint_files) == self.world_size, "Could not find {} files".format(self.world_size)
    
    def _has_fp16_weights(self, state_dict):
        for k in state_dict.keys():
            if k.endswith('_fp16'):
                return True
        return False
    
    def _split_name(self, name):
        name_split = name.split('_rank_')
        param_name = name_split[0]
        is_fp16 = False
        if(len(name_split)==2):
            if '_fp16' in name_split[1]:
                is_fp16 = True
                horizontal_rank = int(name_split[1].split('_fp16')[0])
            else:
                horizontal_rank = int(name_split[1])
        else:
            horizontal_rank = None
        row_split=True if len(param_name.split('_row'))==2 else False
        column_split=True if len(param_name.split('_column'))==2 else False
        param_name = param_name.split('_row')[0].split('_column')[0]
        param_name = param_name + '_fp16' if is_fp16==True else param_name
        return param_name, horizontal_rank, row_split, column_split

    def _aggregate(self, param_dict):
        sharded_params=set()
        for k,v in param_dict.items():
            param_name, horizontal_rank, row_split, column_split = self._split_name(k)
            assert(row_split and column_split, "Invalid case, both row and column can't be split.")
            axis = 0 if row_split else -1 if column_split else None

            if axis is not None: 
                # parameter is sharded
                sharded_params.add(param_name)
                
                if horizontal_rank == 0 and param_name in self.aggregate_state_dict:
                    # delete stale initializer present in state_dict
                    del(self.aggregate_state_dict[param_name])

                if param_name in self.aggregate_state_dict:
                    # Found a previous shard of the param, concatenate shards ordered by ranks
                    self.aggregate_state_dict[param_name] = torch.cat((self.aggregate_state_dict[param_name], v), axis)
                else:
                    self.aggregate_state_dict[param_name] = v
            
            else:
                if k in sharded_params:
                    # stale initializer that must be ignored
                    continue
                if k in self.aggregate_state_dict:
                    assert (self.aggregate_state_dict[k] == v).all(), "Unsharded params must have the same value"
                else:
                    self.aggregate_state_dict[k] = v

    def aggregate_checkpoints(self):
        self.aggregate_state_dict=dict()
        for i in range(len(self.checkpoint_files)):
            checkpoint_name = self.checkpoint_files[i]
            print("Loading state dict from: {}".format(checkpoint_name))
            rank_state_dict = torch.load(checkpoint_name, map_location=torch.device("cpu"))
            
            if 'model' in rank_state_dict:
                rank_state_dict = rank_state_dict['model']
            
            if self.clean_state_dict:
                rank_state_dict = self.clean_state_dict(rank_state_dict)
            
            rank_state_dict = split_state_dict(rank_state_dict)

            self._aggregate(rank_state_dict['fp16_param'])
            #need to debug
            # self._aggregate(rank_state_dict['fp32_param'])
            self._aggregate(rank_state_dict['optimizer'])           

        return self.aggregate_state_dict


def megatron_test_4H():
    checkpoint_dir="/bert_ort/aibhanda/faireseq_t5/checkpoints/FP32_D1_H4_P1_1/"
    checkpoint_prefix="ORT_checkpoint"
    pytorch_ckpt_file="pytorch_state.pyt"
    checkpoint_files = list_checkpoint_files(checkpoint_dir, checkpoint_prefix)
    checkpoint_files = sorted(checkpoint_files)
    sd0 = torch.load(checkpoint_files[0], map_location='cpu')['model']
    sd1 = torch.load(checkpoint_files[1], map_location='cpu')['model']
    sd2 = torch.load(checkpoint_files[2], map_location='cpu')['model']
    sd3 = torch.load(checkpoint_files[3], map_location='cpu')['model']

    # pyt_sd = torch.load(os.path.join(checkpoint_dir, pytorch_ckpt_file), map_location='cpu')
    ckpt_agg = CombineMegatronCheckpoint(checkpoint_files)
    agg_sd = ckpt_agg.aggregate_checkpoints()

    # missing_keys = ['decoder.t5_stack.embed_tokens.weight', 'decoder.embed_tokens.weight']
    verified_keys = 0
    for k, v in pyt_sd.items():
        try:
            assert(
                v.size() == agg_sd[k.split('model_.')[1]].size())
            verified_keys += 1
        except:
            print('Skipping: '+k)
   
    return 1

def megatron_test_before_after():
    checkpoint_dir_before="/bert_ort/aibhanda/faireseq_t5/checkpoints/1step_2H/before/"
    # checkpoint_dir_before="/bert_ort/aibhanda/faireseq_t5/checkpoints/1step_2D/before/"
    checkpoint_prefix="ORT_checkpoint"
    pytorch_ckpt_file="pytorch_state.pyt"
    checkpoint_files = list_checkpoint_files(checkpoint_dir_before, checkpoint_prefix)
    checkpoint_files = sorted(checkpoint_files)
    sd0_before = torch.load(checkpoint_files[0], map_location='cpu')['model']
    sd1_before = torch.load(checkpoint_files[1], map_location='cpu')['model']

    checkpoint_dir_after="/bert_ort/aibhanda/faireseq_t5/checkpoints/1step_2H/after/"
    # checkpoint_dir_after="/bert_ort/aibhanda/faireseq_t5/checkpoints/1step_2D/after/"
    checkpoint_prefix="ORT_checkpoint"
    pytorch_ckpt_file="pytorch_state.pyt"
    checkpoint_files = list_checkpoint_files(checkpoint_dir_after, checkpoint_prefix)
    checkpoint_files = sorted(checkpoint_files)
    sd0_after = torch.load(checkpoint_files[0], map_location='cpu')['model']
    sd1_after = torch.load(checkpoint_files[1], map_location='cpu')['model']

    before_keys = set(sd0_before.keys()).intersection(set(sd1_before.keys()))
    after_keys = set(sd0_after.keys()).intersection(set(sd1_after.keys()))

    before_mismatch=[]
    for k in before_keys:
        if not is_equal_tensor(sd0_before[k], sd1_before[k]):
            before_mismatch.append(k)
    print(f"Mismatched {len(before_mismatch)} out of {len(before_keys)}, total keys {len(sd0_before)}, {len(sd1_before)}")
    after_mismatch=[]
    for k in after_keys:
        if not is_equal_tensor(sd0_after[k], sd1_after[k]):
            after_mismatch.append(k)
    print(f"Mismatched {len(after_mismatch)} out of {len(after_keys)}, total keys {len(sd0_after)}, {len(sd1_after)}")
    return 1

def megatron_test_2H():
    checkpoint_dir="/bert_ort/aibhanda/faireseq_t5/checkpoints/FP32_D1_H2_P1_1/"
    checkpoint_prefix="ORT_checkpoint"
    pytorch_ckpt_file="pytorch_state.pyt"
    checkpoint_files = list_checkpoint_files(checkpoint_dir, checkpoint_prefix)
    checkpoint_files = sorted(checkpoint_files)
    sd0 = torch.load(checkpoint_files[0], map_location='cpu')['model']
    sd1 = torch.load(checkpoint_files[1], map_location='cpu')['model']

    # pyt_sd = torch.load(os.path.join(checkpoint_dir, pytorch_ckpt_file), map_location='cpu')
    ckpt_agg = CombineMegatronCheckpoint(checkpoint_files)
    agg_sd = ckpt_agg.aggregate_checkpoints()

    # missing_keys = ['decoder.t5_stack.embed_tokens.weight', 'decoder.embed_tokens.weight']
    verified_keys = 0
    for k, v in pyt_sd.items():
        try:
            assert(
                v.size() == agg_sd[k.split('model_.')[1]].size())
            verified_keys += 1
        except:
            print('Skipping: '+k)
   
    return 1

def megatron_test_1():
    checkpoint_dir="/bert_ort/aibhanda/faireseq_t5/checkpoints/FP32_D1_H4_P1/"
    checkpoint_prefix="ORT_checkpoint"
    pytorch_ckpt_file="pytorch_state.pyt"
    checkpoint_files = list_checkpoint_files(checkpoint_dir, checkpoint_prefix)

    ckpt_agg = CombineMegatronCheckpoint(checkpoint_files)
    aggregate_state_dict = ckpt_agg.aggregate_checkpoints()

    pytorch_state = torch.load(os.path.join(checkpoint_dir, pytorch_ckpt_file), map_location='cpu')
    # missing_keys = ['decoder.t5_stack.embed_tokens.weight', 'decoder.embed_tokens.weight']
    verified_keys = 0
    for k,v in pytorch_state.items():
        try:
            assert(v.size() == aggregate_state_dict[k.split('model_.')[1]].size())
            verified_keys +=1
        except:
            print('Skipping: '+k)
    
    return 1

def megatron_test_2():
    # checkpoint_dir="/bert_ort/aibhanda/faireseq_t5/checkpoints/FP16_D2_H2_P1/"
    checkpoint_dir="/bert_ort/aibhanda/faireseq_t5/checkpoints/FP16_D4_H1_P1/"
    checkpoint_prefix="ORT_checkpoint"
    pytorch_ckpt_file="pytorch_state.pyt"
    checkpoint_files = list_checkpoint_files(checkpoint_dir, checkpoint_prefix)

    sd0 = torch.load(checkpoint_files[0], map_location='cpu')['model']
    sd1 = torch.load(checkpoint_files[1], map_location='cpu')['model']
    sd2 = torch.load(checkpoint_files[2], map_location='cpu')['model']
    sd3 = torch.load(checkpoint_files[3], map_location='cpu')['model']

    assert sd0.keys() == sd2.keys()
    assert sd1.keys() == sd3.keys() 

    for k in sd0.keys():
        assert (sd0[k] == sd2[k]).all()
    
    for k in sd1.keys():
        assert (sd1[k] == sd3[k]).all()
   
    return 1

def megatron_test_Z_H():
    # checkpoint_dir="/bert_ort/aibhanda/faireseq_t5/checkpoints/FP16_D2_H2_P1/"
    checkpoint_dir="/bert_ort/aibhanda/faireseq_t5/checkpoints/FP32_D2_H2_P1_zero/"
    checkpoint_prefix="ORT_checkpoint"
    pytorch_ckpt_file="pytorch_state.pyt"
    checkpoint_files = list_checkpoint_files(checkpoint_dir, checkpoint_prefix)
    checkpoint_files = sorted(checkpoint_files)

    sd0 = torch.load(checkpoint_files[0], map_location='cpu')['model']
    sd1 = torch.load(checkpoint_files[1], map_location='cpu')['model']
    sd2 = torch.load(checkpoint_files[2], map_location='cpu')['model']
    sd3 = torch.load(checkpoint_files[3], map_location='cpu')['model']

    ssd0 = split_state_dict(sd0)
    ssd1 = split_state_dict(sd1)
    ssd2 = split_state_dict(sd2)
    ssd3 = split_state_dict(sd3)
    
    assert is_equal(ssd0['fp32_param'], ssd2['fp32_param'])
    assert is_equal(ssd1['fp32_param'], ssd3['fp32_param'])

    return 1

def megatron_test_3():
    horizontal_parallel_size = 4
    world_size = 4
    data_parallel_size = int(world_size/horizontal_parallel_size)

    num_data_groups = int(world_size / data_parallel_size)
    for data_group_id in range(num_data_groups):
        data_group_ranks=[]
        for r in range(data_parallel_size):
            data_group_ranks.append(data_group_id + horizontal_parallel_size * r)
        print("Data Group: {} : {}".format(data_group_id, data_group_ranks))
    
    # for world_rank in range(world_size):
    #     print("Rank: {}".format(world_rank))
    #     data_group_id = world_rank % horizontal_parallel_size
    #     rank_in_owning_data_group = world_rank / horizontal_parallel_size
    #     data_group_ranks=[]
    #     for r in range(data_parallel_size):
    #         data_group_ranks.append(data_group_id + horizontal_parallel_size * r)
    #     print("Data Group: {} : {}".format(data_group_id, data_group_ranks))

    
    num_hori_groups = int(world_size / horizontal_parallel_size)
    combined_sd = dict()
    for hori_group_id in range(num_hori_groups):
        hori_group_ranks=[]
        for r in range(horizontal_parallel_size):
            hori_group_ranks.append(hori_group_id * horizontal_parallel_size + r)
        print("Horizntal Group: {} : {}".format(hori_group_id, hori_group_ranks))

        


megatron_test_before_after()
# megatron_test_2H()
# megatron_test_2()
# megatron_test_Z_H()
# megatron_test_3()
