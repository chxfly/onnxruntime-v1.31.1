// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace OrtApis {

ORT_API(const OrtApi*, GetApi, uint32_t version);
ORT_API(const char*, GetVersionString);

ORT_API(void, ReleaseEnv, OrtEnv*);
ORT_API(void, ReleaseStatus, OrtStatus*);
ORT_API(void, ReleaseMemoryInfo, OrtMemoryInfo*);
ORT_API(void, ReleaseSession, OrtSession*);
ORT_API(void, ReleaseValue, OrtValue*);
ORT_API(void, ReleaseRunOptions, OrtRunOptions*);
ORT_API(void, ReleaseTypeInfo, OrtTypeInfo*);
ORT_API(void, ReleaseTensorTypeAndShapeInfo, OrtTensorTypeAndShapeInfo*);
ORT_API(void, ReleaseSessionOptions, OrtSessionOptions*);
ORT_API(void, ReleaseCustomOpDomain, OrtCustomOpDomain*);

ORT_API_STATUS_IMPL(CreateStatus, OrtErrorCode code, _In_ const char* msg);
OrtErrorCode ORT_API_CALL GetErrorCode(_In_ const OrtStatus* status) NO_EXCEPTION ORT_ALL_ARGS_NONNULL;
const char* ORT_API_CALL GetErrorMessage(_In_ const OrtStatus* status) NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

ORT_API_STATUS_IMPL(CreateEnv, OrtLoggingLevel default_logging_level, _In_ const char* logid, _Outptr_ OrtEnv** out)
ORT_ALL_ARGS_NONNULL;
ORT_API_STATUS_IMPL(CreateEnvWithCustomLogger, OrtLoggingFunction logging_function, _In_opt_ void* logger_param, OrtLoggingLevel default_warning_level, _In_ const char* logid, _Outptr_ OrtEnv** out);

ORT_API_STATUS_IMPL(CreateSession, _In_ const OrtEnv* env, _In_ const ORTCHAR_T* model_path,
                    _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out);

ORT_API_STATUS_IMPL(CreateSessionFromArray, _In_ const OrtEnv* env, _In_ const void* model_data, size_t model_data_length,
                    _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out);

ORT_API_STATUS_IMPL(Run, _Inout_ OrtSession* sess,
                    _In_opt_ const OrtRunOptions* run_options,
                    _In_ const char* const* input_names, _In_ const OrtValue* const* input, size_t input_len,
                    _In_ const char* const* output_names, size_t output_names_len, _Outptr_ OrtValue** output);

ORT_API_STATUS_IMPL(CreateSessionOptions, OrtSessionOptions** out);
ORT_API_STATUS_IMPL(CloneSessionOptions, const OrtSessionOptions* input, OrtSessionOptions** out);
ORT_API_STATUS_IMPL(EnableSequentialExecution, _In_ OrtSessionOptions* options);
ORT_API_STATUS_IMPL(DisableSequentialExecution, _In_ OrtSessionOptions* options);
ORT_API_STATUS_IMPL(SetOptimizedModelFilePath, _In_ OrtSessionOptions* options, _In_ const ORTCHAR_T* optimized_model_filepath);
ORT_API_STATUS_IMPL(EnableProfiling, _In_ OrtSessionOptions* options, _In_ const ORTCHAR_T* profile_file_prefix);
ORT_API_STATUS_IMPL(DisableProfiling, _In_ OrtSessionOptions* options);
ORT_API_STATUS_IMPL(EnableMemPattern, _In_ OrtSessionOptions* options);
ORT_API_STATUS_IMPL(DisableMemPattern, _In_ OrtSessionOptions* options);
ORT_API_STATUS_IMPL(EnableCpuMemArena, _In_ OrtSessionOptions* options);
ORT_API_STATUS_IMPL(DisableCpuMemArena, _In_ OrtSessionOptions* options);
ORT_API_STATUS_IMPL(SetSessionLogId, _In_ OrtSessionOptions* options, const char* logid);
ORT_API_STATUS_IMPL(SetSessionLogVerbosityLevel, _In_ OrtSessionOptions* options, int session_log_verbosity_level);
ORT_API_STATUS_IMPL(SetSessionLogSeverityLevel, _In_ OrtSessionOptions* options, int session_log_severity_level);
ORT_API_STATUS_IMPL(SetSessionGraphOptimizationLevel, _In_ OrtSessionOptions* options,
                    GraphOptimizationLevel graph_optimization_level);
ORT_API_STATUS_IMPL(SetIntraOpNumThreads, _Inout_ OrtSessionOptions* options, int intra_op_num_threads);
ORT_API_STATUS_IMPL(SetInterOpNumThreads, _Inout_ OrtSessionOptions* options, int inter_op_num_threads);

ORT_API_STATUS_IMPL(CreateCustomOpDomain, _In_ const char* domain, _Outptr_ OrtCustomOpDomain** out);
ORT_API_STATUS_IMPL(CustomOpDomain_Add, _Inout_ OrtCustomOpDomain* custom_op_domain, _In_ OrtCustomOp* op);
ORT_API_STATUS_IMPL(AddCustomOpDomain, _Inout_ OrtSessionOptions* options, _In_ OrtCustomOpDomain* custom_op_domain);
ORT_API_STATUS_IMPL(RegisterCustomOpsLibrary, _Inout_ OrtSessionOptions* options, _In_ const char* library_path, void** library_handle);

ORT_API_STATUS_IMPL(SessionGetInputCount, _In_ const OrtSession* sess, _Out_ size_t* out);
ORT_API_STATUS_IMPL(SessionGetOutputCount, _In_ const OrtSession* sess, _Out_ size_t* out);
ORT_API_STATUS_IMPL(SessionGetOverridableInitializerCount, _In_ const OrtSession* sess, _Out_ size_t* out);
ORT_API_STATUS_IMPL(SessionGetInputTypeInfo, _In_ const OrtSession* sess, size_t index, _Outptr_ OrtTypeInfo** type_info);
ORT_API_STATUS_IMPL(SessionGetOutputTypeInfo, _In_ const OrtSession* sess, size_t index, _Outptr_ OrtTypeInfo** type_info);
ORT_API_STATUS_IMPL(SessionGetOverridableInitializerTypeInfo, _In_ const OrtSession* sess, size_t index, _Outptr_ OrtTypeInfo** type_info);
ORT_API_STATUS_IMPL(SessionGetInputName, _In_ const OrtSession* sess, size_t index, _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
ORT_API_STATUS_IMPL(SessionGetOutputName, _In_ const OrtSession* sess, size_t index, _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
ORT_API_STATUS_IMPL(SessionGetOverridableInitializerName, _In_ const OrtSession* sess, size_t index,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value);

ORT_API_STATUS_IMPL(CreateRunOptions, _Outptr_ OrtRunOptions** out);

ORT_API_STATUS_IMPL(RunOptionsSetRunLogVerbosityLevel, _Inout_ OrtRunOptions* options, int value);
ORT_API_STATUS_IMPL(RunOptionsSetRunLogSeverityLevel, _Inout_ OrtRunOptions* options, int value);
ORT_API_STATUS_IMPL(RunOptionsSetRunTag, _In_ OrtRunOptions*, _In_ const char* run_tag);

ORT_API_STATUS_IMPL(RunOptionsGetRunLogVerbosityLevel, _In_ const OrtRunOptions* options, _Out_ int* out);
ORT_API_STATUS_IMPL(RunOptionsGetRunLogSeverityLevel, _In_ const OrtRunOptions* options, _Out_ int* out);
ORT_API_STATUS_IMPL(RunOptionsGetRunTag, _In_ const OrtRunOptions*, _Out_ const char** out);

ORT_API_STATUS_IMPL(RunOptionsSetTerminate, _Inout_ OrtRunOptions* options);
ORT_API_STATUS_IMPL(RunOptionsUnsetTerminate, _Inout_ OrtRunOptions* options);

ORT_API_STATUS_IMPL(CreateTensorAsOrtValue, _Inout_ OrtAllocator* allocator,
                    _In_ const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type,
                    _Outptr_ OrtValue** out);
ORT_API_STATUS_IMPL(CreateTensorWithDataAsOrtValue, _In_ const OrtMemoryInfo* info,
                    _Inout_ void* p_data, size_t p_data_len, _In_ const int64_t* shape, size_t shape_len,
                    ONNXTensorElementDataType type, _Outptr_ OrtValue** out);
ORT_API_STATUS_IMPL(IsTensor, _In_ const OrtValue* value, _Out_ int* out);
ORT_API_STATUS_IMPL(GetTensorMutableData, _Inout_ OrtValue* value, _Outptr_ void** out);
ORT_API_STATUS_IMPL(FillStringTensor, _Inout_ OrtValue* value, _In_ const char* const* s, size_t s_len);
ORT_API_STATUS_IMPL(GetStringTensorDataLength, _In_ const OrtValue* value, _Out_ size_t* len);
ORT_API_STATUS_IMPL(GetStringTensorContent, _In_ const OrtValue* value, _Out_ void* s, size_t s_len,
                    _Out_ size_t* offsets, size_t offsets_len);
ORT_API_STATUS_IMPL(CastTypeInfoToTensorInfo, _In_ const OrtTypeInfo*, _Out_ const OrtTensorTypeAndShapeInfo** out);
ORT_API_STATUS_IMPL(GetOnnxTypeFromTypeInfo, _In_ const OrtTypeInfo*, _Out_ enum ONNXType* out);
ORT_API_STATUS_IMPL(CreateTensorTypeAndShapeInfo, _Outptr_ OrtTensorTypeAndShapeInfo** out);
ORT_API_STATUS_IMPL(SetTensorElementType, _Inout_ OrtTensorTypeAndShapeInfo*, enum ONNXTensorElementDataType type);
ORT_API_STATUS_IMPL(SetDimensions, OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count);
ORT_API_STATUS_IMPL(GetTensorElementType, _In_ const OrtTensorTypeAndShapeInfo*, _Out_ enum ONNXTensorElementDataType* out);
ORT_API_STATUS_IMPL(GetDimensionsCount, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ size_t* out);
ORT_API_STATUS_IMPL(GetDimensions, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values, size_t dim_values_length);
ORT_API_STATUS_IMPL(GetTensorShapeElementCount, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ size_t* out);
ORT_API_STATUS_IMPL(GetTensorTypeAndShape, _In_ const OrtValue* value, _Outptr_ OrtTensorTypeAndShapeInfo** out);
ORT_API_STATUS_IMPL(GetTypeInfo, _In_ const OrtValue* value, _Outptr_ OrtTypeInfo** out);
ORT_API_STATUS_IMPL(GetValueType, _In_ const OrtValue* value, _Out_ enum ONNXType* out);
ORT_API_STATUS_IMPL(AddFreeDimensionOverride, _Inout_ OrtSessionOptions* options, _In_ const char* symbolic_dim, _In_ int64_t dim_override);

ORT_API_STATUS_IMPL(CreateMemoryInfo, _In_ const char* name1, enum OrtAllocatorType type, int id1, enum OrtMemType mem_type1, _Outptr_ OrtMemoryInfo** out);
ORT_API_STATUS_IMPL(CreateCpuMemoryInfo, enum OrtAllocatorType type, enum OrtMemType mem_type1, _Outptr_ OrtMemoryInfo** out)
ORT_ALL_ARGS_NONNULL;
ORT_API_STATUS_IMPL(CompareMemoryInfo, _In_ const OrtMemoryInfo* info1, _In_ const OrtMemoryInfo* info2, _Out_ int* out)
ORT_ALL_ARGS_NONNULL;
ORT_API_STATUS_IMPL(MemoryInfoGetName, _In_ const OrtMemoryInfo* ptr, _Out_ const char** out);
ORT_API_STATUS_IMPL(MemoryInfoGetId, _In_ const OrtMemoryInfo* ptr, _Out_ int* out);
ORT_API_STATUS_IMPL(MemoryInfoGetMemType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtMemType* out);
ORT_API_STATUS_IMPL(MemoryInfoGetType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtAllocatorType* out);

ORT_API_STATUS_IMPL(AllocatorAlloc, _Inout_ OrtAllocator* ptr, size_t size, _Outptr_ void** out);
ORT_API_STATUS_IMPL(AllocatorFree, _Inout_ OrtAllocator* ptr, void* p);
ORT_API_STATUS_IMPL(AllocatorGetInfo, _In_ const OrtAllocator* ptr, _Out_ const OrtMemoryInfo** out);
ORT_API_STATUS_IMPL(GetAllocatorWithDefaultOptions, _Outptr_ OrtAllocator** out);
ORT_API_STATUS_IMPL(GetValue, _In_ const OrtValue* value, int index, _Inout_ OrtAllocator* allocator, _Outptr_ OrtValue** out);
ORT_API_STATUS_IMPL(GetValueCount, _In_ const OrtValue* value, _Out_ size_t* out);
ORT_API_STATUS_IMPL(CreateValue, _In_ const OrtValue* const* in, size_t num_values, enum ONNXType value_type,
                    _Outptr_ OrtValue** out);
ORT_API_STATUS_IMPL(CreateOpaqueValue, _In_ const char* domain_name, _In_ const char* type_name,
                    _In_ const void* data_container, size_t data_container_size, _Outptr_ OrtValue** out);
ORT_API_STATUS_IMPL(GetOpaqueValue, _In_ const char* domain_name, _In_ const char* type_name,
                    _In_ const OrtValue* in, _Out_ void* data_container, size_t data_container_size);

ORT_API_STATUS_IMPL(KernelInfoGetAttribute_float, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ float* out);
ORT_API_STATUS_IMPL(KernelInfoGetAttribute_int64, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ int64_t* out);
ORT_API_STATUS_IMPL(KernelInfoGetAttribute_string, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ char* out, _Inout_ size_t* size);

ORT_API_STATUS_IMPL(KernelContext_GetInputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out);
ORT_API_STATUS_IMPL(KernelContext_GetOutputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out);
ORT_API_STATUS_IMPL(KernelContext_GetInput, _In_ const OrtKernelContext* context, _In_ size_t index, _Out_ const OrtValue** out);
ORT_API_STATUS_IMPL(KernelContext_GetOutput, _Inout_ OrtKernelContext* context, _In_ size_t index, _In_ const int64_t* dim_values, size_t dim_count, _Out_ OrtValue** out);

}  // namespace OrtApis
