__global__ void blurMain(unsigned int w, unsigned int r, unsigned int * src, unsigned int * output);
__global__ void blurBottomEdge(unsigned int w, unsigned int h, unsigned int r, unsigned int * src, unsigned int * output);

__global__ void divideMain(unsigned int w, unsigned int h, unsigned int r, unsigned int * src, unsigned char * output);
__global__ void divideTopEdge(unsigned int w, unsigned int r, unsigned int * src, unsigned char * output);
__global__ void divideBottomEdge(unsigned int w, unsigned int h, unsigned int r, unsigned int * src, unsigned char * output);