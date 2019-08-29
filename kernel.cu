#include<stdio.h>
#include "ppmFile.c"
#include "blur.cuh"

void copy(unsigned char * src, unsigned int * dst, unsigned int w, unsigned int h) {
	h = w * h * 3;
	for (unsigned int index = 0; index < h; index++) {
		dst[index] = (unsigned int)src[index];
	}
}

void transpose(unsigned int * src, unsigned int * dst, unsigned int w, unsigned int h) {
	for (unsigned int y = 0; y < h; y++) {
		for (unsigned int x = 0; x < w; x++) {
			dst[x*h * 3 + y * 3] = src[x * 3 + y * w * 3];
			dst[x*h * 3 + y * 3 + 1] = src[x * 3 + y * w * 3 + 1];
			dst[x*h * 3 + y * 3 + 2] = src[x * 3 + y * w * 3 + 2];
		}
	}
}

void transposeChar(unsigned char * src, unsigned char * dst, unsigned int w, unsigned int h) {
	for (unsigned int y = 0; y < h; y++) {
		for (unsigned int x = 0; x < w; x++) {
			dst[x*h * 3 + y * 3] = src[x * 3 + y * w * 3];
			dst[x*h * 3 + y * 3 + 1] = src[x * 3 + y * w * 3 + 1];
			dst[x*h * 3 + y * 3 + 2] = src[x * 3 + y * w * 3 + 2];
		}
	}
}

int main(int argc, char* argv[]) {
	Image *src, *dest; //source and destination images
	int w, h; //width and height of image

	if (argc < 4)
		return -1;

	int r = atoi(argv[1]);
	char *inputFile = argv[2];
	char *outputFile = argv[3];

	if (r < 0)
		return -1;

	src = ImageRead(inputFile);
	w = ImageWidth(src);
	h = ImageHeight(src);
	dest = ImageCreate(w, h);
	printf("%dx%d, %d\n", w, h, r);

	unsigned char *data, *finalImage;
	unsigned int *srcData, *destData = NULL, *destData2, *dataTr, *dataTest;

	cudaMalloc((void**)&srcData, w * h * 3 * sizeof(unsigned int));
	cudaMalloc((void**)&destData, w * h * 3 * sizeof(unsigned int));
	cudaMalloc((void**)&destData2, w * h * 3 * sizeof(unsigned int));
	cudaMalloc((void**)&finalImage, w * h * 3 * sizeof(unsigned char));
	data = (unsigned char*)malloc(w * h * 3 * sizeof(unsigned char));
	dataTest = (unsigned int*)malloc(w * h * 3 * sizeof(unsigned int));
	dataTr = (unsigned int*)malloc(w * h * 3 * sizeof(unsigned int));

	cudaMemset(destData, 0, w*h * 3 * sizeof(unsigned int));

	copy(src->data, dataTest, w, h);
	cudaMemcpy(srcData, dataTest, w * h * 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);

	unsigned int tmp = h / 32, tmp2 = h - tmp * 32;
	blurMain << < tmp, 32 >> > (w, r, srcData, destData);
	cudaDeviceSynchronize();
	blurBottomEdge << < 1, tmp2 >> > (w, tmp * 32, r, srcData, destData);
	cudaDeviceSynchronize();

	cudaMemcpy(dataTest, destData, w * h * 3 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	transpose(dataTest, dataTr, w, h);
	cudaMemcpy(destData, dataTr, w * h * 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemset(destData2, 0, w * h * 3 * sizeof(unsigned int));

	tmp = w / 32;
	tmp2 = w - tmp * 32;
	blurMain << < tmp, 32 >> > (h, r, destData, destData2);
	cudaDeviceSynchronize();
	blurBottomEdge << < 1, tmp2 >> > (h, tmp * 32, r, destData, destData2);
	cudaDeviceSynchronize();

	//start dividing edges from # of pixels
	divideTopEdge << < 1, r >> > (h, r, destData2, finalImage);
	cudaDeviceSynchronize();
	divideBottomEdge << < 1, r >> > (h, w - r, r, destData2, finalImage);
	cudaDeviceSynchronize();

	//start dividing the main area from # of pixels
	tmp = (w - 2 * r) / 32;
	tmp2 = (w - 2 * r) - tmp * 32;
	divideMain << < tmp, 32 >> > (h, 0, r, destData2, finalImage);
	cudaDeviceSynchronize();
	divideMain << < 1, tmp2 >> > (h, w - 2 * r - tmp2, r, destData2, finalImage);
	cudaDeviceSynchronize();

	cudaMemcpy(data, finalImage, w * h * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	transposeChar(data, dest->data, h, w);

	ImageWrite(dest, outputFile);

	cudaFree(srcData);
	cudaFree(destData);
	cudaFree(destData2);
	cudaFree(finalImage);
	free(data);
	free(dataTr);
	free(dataTest);

	return 0;
}