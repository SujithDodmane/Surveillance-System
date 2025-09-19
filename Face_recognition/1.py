import dlib
print(dlib.DLIB_USE_CUDA)   # True if built with CUDA
print(dlib.cuda.get_num_devices())  # >0 if GPU is available
