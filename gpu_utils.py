import numpy as np
import subprocess

class GPUMemoryUtils:
    def get_used_gpu_memory(self):
        """
        Adapted code from mjstevens777
        https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/3

        Get the current GPU usage.

        Return:
            gpu_memory: numpy array
                memory usage as integers in MB.
        """
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8')

        # Convert lines into list
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        gpu_memory = np.array(gpu_memory)

        return gpu_memory


    def get_free_gpus(self, memory_threshold=150):
        """
        Get indices of free GPUs.

        A GPU is free if its used memory is less than memory_threshold

        Argument:
            memory_threshold: int
        Return:
            free_gpus: numpy array
                indices of free GPUs
        """
        used_gpu_memory = self.get_used_gpu_memory()
        free_gpus = np.flatnonzero(used_gpu_memory < memory_threshold)

        return free_gpus


    def get_num_free_gpus(self, memory_threshold=150):
        """
        Get number of available GPUs

        Argument:
            memory_threshold: int
        Return:
            num_free_gpus: int
                number of free GPUs
        """
        used_gpu_memory = self.get_used_gpu_memory()
        num_free_gpus = np.sum(used_gpu_memory < memory_threshold)

        return num_free_gpus