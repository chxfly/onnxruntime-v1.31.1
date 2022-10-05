import argparse
import logging
import os
import shutil
import sys

import coloredlogs
import onnx

logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from optimizer import optimize_model  # noqa: E402


def optimize_stable_diffusion_onnx(source_dir, target_dir, float16):
    """Load stable diffusion onnx models from source directory,
    optimize graph and convert from fp32 to fp16,
    then save models to target directory"""
    for name in ["unet", "vae_encoder", "vae_decoder", "text_encoder", "safety_checker"]:
        onnx_model_path = os.path.join(source_dir, name, "model.onnx")
        print(f"processing {onnx_model_path} ...")

        # The following will fuse Attention, LayerNormalization and Gelu.
        # Do it before fp16 conversion, otherwise they cannot be fused later.
        # Right now, onnxruntime does not save >2GB model so we use script to optimize unet instead.
        m = optimize_model(
            onnx_model_path,
            model_type="unet",
            num_heads=0,
            hidden_size=0,
            opt_level=0,
            optimization_options=None,
            use_gpu=False,
        )

        if float16:
            # Use op_bloack_list to force some operators to compute in FP32.
            # TODO: might need some tuning to trade-off performance and accuracy.
            if name == "safety_checker":
                m.convert_float_to_float16(op_block_list=["Where"])
            else:
                m.convert_float_to_float16()

        # Overwrite existing models when source_dir==target_dir. Otherwise, you might need copy non-onnx files manually.
        optimized_model_path = os.path.join(target_dir, name, "model.onnx")
        output_dir = os.path.dirname(optimized_model_path)
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        onnx.save_model(m.model, optimized_model_path)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Optimization tool for stable diffusion onnx pipeline")

    parser.add_argument("--input", required=True, type=str, help="input onnx pipeline directory")

    parser.add_argument("--output", required=True, type=str, help="output onnx pipeline directory")

    parser.add_argument(
        "--float16",
        required=False,
        action="store_true",
        help="Convert models to float16",
    )
    parser.set_defaults(float16=False)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    coloredlogs.install(fmt="%(funcName)20s: %(message)s")
    optimize_stable_diffusion_onnx(args.input, args.output, args.float16)


if __name__ == "__main__":
    main()
