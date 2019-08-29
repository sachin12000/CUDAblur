#include "blur.cuh"

__global__ void blurMain(unsigned int w, unsigned int r, unsigned int * src, unsigned int * output) {
	unsigned int offset = ((32 * blockIdx.x + threadIdx.x) * w) * 3;
	unsigned int endIndex = offset + r * 3;

	//first pixel of the row
	for (unsigned int index = offset; index <= endIndex; index += 3) {
		output[offset] += src[index];
		output[offset + 1] += src[index + 1];
		output[offset + 2] += src[index + 2];
	}

	//deal with pxiels with x < r+1
	for (unsigned int index = offset + 3; index <= endIndex; index += 3) {
		output[index] = output[index - 3] + src[index + r * 3];
		output[index + 1] = output[index - 3 + 1] + src[index + r * 3 + 1];
		output[index + 2] = output[index - 3 + 2] + src[index + r * 3 + 2];
	}

	endIndex = offset + (w - r) * 3;
	for (unsigned int index = offset + (r + 1) * 3; index < endIndex; index += 3) {
		output[index] = output[index - 3];
		output[index + 1] = output[index - 3 + 1];
		output[index + 2] = output[index - 3 + 2];

		output[index] += src[index + (r + 1) * 3];
		output[index + 1] += src[index + (r + 1) * 3 + 1];
		output[index + 2] += src[index + (r + 1) * 3 + 2];

		output[index] -= src[index - r * 3];
		output[index + 1] -= src[index - r * 3 + 1];
		output[index + 2] -= src[index - r * 3 + 2];
	}

	endIndex = offset + w * 3;
	for (unsigned int index = offset + (w - r) * 3; index < endIndex; index += 3) {
		output[index] = output[index - 3] - src[index - r * 3];
		output[index + 1] = output[index - 3 + 1] - src[index - r * 3 + 1];
		output[index + 2] = output[index - 3 + 2] - src[index - r * 3 + 2];
	}
}

__global__ void blurBottomEdge(unsigned int w, unsigned int h, unsigned int r, unsigned int * src, unsigned int * output) {
	unsigned int offset = (h + threadIdx.x) * w * 3;
	unsigned int endIndex = offset + r * 3;

	//first pixel of the row
	for (unsigned int index = offset; index <= endIndex; index += 3) {
		output[offset] += src[index];
		output[offset + 1] += src[index + 1];
		output[offset + 2] += src[index + 2];
	}

	//deal with pxiels with x < r+1
	for (unsigned int index = offset + 3; index <= endIndex; index += 3) {
		output[index] = output[index - 3] + src[index + r * 3];
		output[index + 1] = output[index - 3 + 1] + src[index + r * 3 + 1];
		output[index + 2] = output[index - 3 + 2] + src[index + r * 3 + 2];
	}

	endIndex = offset + (w - r) * 3;
	for (unsigned int index = offset + (r + 1) * 3; index < endIndex; index += 3) {
		output[index] = output[index - 3];
		output[index + 1] = output[index - 3 + 1];
		output[index + 2] = output[index - 3 + 2];

		output[index] += src[index + (r + 1) * 3];
		output[index + 1] += src[index + (r + 1) * 3 + 1];
		output[index + 2] += src[index + (r + 1) * 3 + 2];

		output[index] -= src[index - r * 3];
		output[index + 1] -= src[index - r * 3 + 1];
		output[index + 2] -= src[index - r * 3 + 2];
	}

	endIndex = offset + w * 3;
	for (unsigned int index = offset + (w - r) * 3; index < endIndex; index += 3) {
		output[index] = output[index - 3] - src[index - r * 3];
		output[index + 1] = output[index - 3 + 1] - src[index - r * 3 + 1];
		output[index + 2] = output[index - 3 + 2] - src[index - r * 3 + 2];
	}
}

__global__ void divideMain(unsigned int w, unsigned int h, unsigned int r, unsigned int * src, unsigned char * output) {
	unsigned int index, offset = (blockIdx.x * 32 + threadIdx.x + h + r) * w * 3;
	float height = 2 * r + 1;
	float divider = (r + 1)*height;

	for (index = offset, offset += r * 3; index < offset; index += 3) {
		output[index] = (unsigned char)(src[index] / divider);
		output[index + 1] = (unsigned char)(src[index + 1] / divider);
		output[index + 2] = (unsigned char)(src[index + 2] / divider);
		divider += height;
	}

	for (index = offset, offset += (w - 2 * r) * 3; index < offset; index += 3) {
		output[index] = (unsigned char)(src[index] / divider);
		output[index + 1] = (unsigned char)(src[index + 1] / divider);
		output[index + 2] = (unsigned char)(src[index + 2] / divider);
	}

	for (index = offset, offset += r * 3; index < offset; index += 3) {
		divider -= height;
		output[index] = (unsigned char)(src[index] / divider);
		output[index + 1] = (unsigned char)(src[index + 1] / divider);
		output[index + 2] = (unsigned char)(src[index + 2] / divider);
	}
}

__global__ void divideTopEdge(unsigned int w, unsigned int r, unsigned int * src, unsigned char * output) {
	unsigned int index, offset = threadIdx.x * w * 3;
	float height = r + 1 + threadIdx.x;
	float divider = (r + 1)*height;

	for (index = offset, offset += r * 3; index < offset; index += 3) {
		output[index] = (unsigned char)(src[index] / divider);
		output[index + 1] = (unsigned char)(src[index + 1] / divider);
		output[index + 2] = (unsigned char)(src[index + 2] / divider);
		divider += height;
	}

	for (index = offset, offset += (w - 2 * r) * 3; index < offset; index += 3) {
		output[index] = (unsigned char)(src[index] / divider);
		output[index + 1] = (unsigned char)(src[index + 1] / divider);
		output[index + 2] = (unsigned char)(src[index + 2] / divider);
	}

	for (index = offset, offset += r * 3; index < offset; index += 3) {
		divider -= height;
		output[index] = (unsigned char)(src[index] / divider);
		output[index + 1] = (unsigned char)(src[index + 1] / divider);
		output[index + 2] = (unsigned char)(src[index + 2] / divider);
	}
}

__global__ void divideBottomEdge(unsigned int w, unsigned int h, unsigned int r, unsigned int * src, unsigned char * output) {
	unsigned int index, offset = (h + threadIdx.x) * w * 3;
	float height = 2 * r - threadIdx.x;
	float divider = (r + 1)*height;

	for (index = offset, offset += r * 3; index < offset; index += 3) {
		output[index] = (unsigned char)(src[index] / divider);
		output[index + 1] = (unsigned char)(src[index + 1] / divider);
		output[index + 2] = (unsigned char)(src[index + 2] / divider);
		divider += height;
	}

	for (index = offset, offset += (w - 2 * r) * 3; index < offset; index += 3) {
		output[index] = (unsigned char)(src[index] / divider);
		output[index + 1] = (unsigned char)(src[index + 1] / divider);
		output[index + 2] = (unsigned char)(src[index + 2] / divider);
	}

	for (index = offset, offset += r * 3; index < offset; index += 3) {
		divider -= height;
		output[index] = (unsigned char)(src[index] / divider);
		output[index + 1] = (unsigned char)(src[index + 1] / divider);
		output[index + 2] = (unsigned char)(src[index + 2] / divider);
	}
}