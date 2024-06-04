float **generate_coeffs_copies_preset(float *rev_coeffs);
float *reverse_filter_coeffs_merge();
void cuda_filter_rt(float *all_data, float *filter_input, float *filter_output);
float *simd_fir_filter_wma(float** filter_1_coeffs, float* all_data, float * filter_input, float* filter_output);
