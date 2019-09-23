import numpy as np
from pycuda.compiler import SourceModule

drv = None#.init()
dev = None#drv.Device(0)
ctx = None#dev.make_context()
mod = None#kernel()


def kernel():
    mod = SourceModule(
    """
    __global__ void filtro_mamalon(float *dest, float *vec, int *dims, float *scale, int *kernel_size){
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int Yi = (x / dims[1]) / (scale[0]*scale[1]);
    	int Xi = (int)(x/scale[0]) % dims[1];

        if(vec[Yi*dims[1] + Xi] > 0){
            dest[x] = vec[Yi*dims[1] + Xi];
            return;
        }

    	float mean = 0.0;
        float den = 0.0;
        float kernel[25];

        int c = 0;
    	for(int i=Yi-(kernel_size[0]/2);i<=Yi+(kernel_size[0]/2);i++){
    		for(int j=Xi-(kernel_size[0]/2);j<=Xi+(kernel_size[0]/2);j++){
    			int index = i*dims[1] + j;
    			if(vec[index] > 0){
    	            mean += vec[index];
                    den += 1.0;
                }
                kernel[c] = vec[index];
                c++;
    		}
    	}

        /*for(int i=1; i<kernel_size[0]; i++){
    	   for(int j=0; j<kernel_size[0]-i; j++){
    			if(kernel[j]>kernel[j+1]){
    			    const float aux    = kernel[j+1];
    				kernel[j+1] = kernel[j];
    				kernel[j]   = aux;
    			}
    		}
    	}*/
        //dest[x] = (kernel[kernel_size[0]/2]+kernel[kernel_size[0]/2 +1])/2;
        //dest[x] = kernel[kernel_size[0]/2];
        dest[x] = mean/den;
    }

    __global__ void projection(float *dest, float *vector, int *h_count, int *v_count, int *lidar_dims, int *len_vect){
        const int i = blockIdx.x * blockDim.x + threadIdx.x;

        //int row = (int)((float)i/(float)lidar_dims[0]);
        dest[v_count[i]*int(lidar_dims[0]) + h_count[i]] = ((vector[len_vect[0] - i*3] - 1.0)/(100.0 - 1.0))*255.0;
    }
    """
    )
    return mod


def median_interpolate_cuda(im, xscale, yscale, kernel_size):

    enlargedImg = np.full((im.shape[0]*yscale, im.shape[1]*xscale), -1)
    rowScale = float(im.shape[0]) / float(enlargedImg.shape[0])
    colScale = float(im.shape[1]) / float(enlargedImg.shape[1])

    dims = np.int32([im.shape[0], im.shape[1]])
    kernel_size = np.int32([kernel_size])
    scale = np.float32([xscale, yscale])
    new_row_size = enlargedImg.shape[0]
    im = im[:, :].ravel().astype(np.float32)
    enlargedImg = enlargedImg[:, :].ravel().astype(np.float32)
    grid_size = int(np.ceil(enlargedImg.shape[0]/1024.0))

    ctx = dev.make_context()
    mod = kernel()
    filtro = mod.get_function("filtro_mamalon")
    filtro(drv.Out(enlargedImg), drv.In(im), drv.In(dims), drv.In(scale), drv.In(kernel_size),
        block=(1024, 1, 1), grid=(grid_size, 1))
    enlargedImg = np.reshape(enlargedImg, (new_row_size, -1))
    ctx.pop()
    return enlargedImg


def projection_cuda(image, vector, h_count, v_count, lidar_dims, new_row_size):
    image = image[:, :].ravel().astype(np.float32)
    grid_size = len(h_count)/1000

    ctx = dev.make_context()
    mod = kernel()
    projection = mod.get_function("projection")
    projection(drv.Out(image), drv.In(vector), drv.In(h_count), drv.In(v_count), drv.In(lidar_dims), drv.In(np.int32([len(h_count)*3])),
        block=(1000, 1, 1), grid=(grid_size, 1))
    image = np.reshape(image, (new_row_size, -1))
    ctx.pop()

    return image


#
