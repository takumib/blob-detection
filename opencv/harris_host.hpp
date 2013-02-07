
extern void wakeup(void);
extern void run_gpu(guchar *host_image, const int width, const int height);
extern void run_gpu_op(guchar *host_image, const int width, const int height);
extern void run_gpu_shared_op(guchar *host_image, const int width, const int height);
extern int has_cuda_device(void);
