#include "utils.h"
struct ExtremeValues{
        float max;
        float min;
};

__global__ 
void findExtremeValues( const float* const d_data, 
                        const size_t numCols,
                       int numRows,
                        ExtremeValues * d_extreme){

//        extern __shared__ s_extreme[];
        // no need to initialize since we are not accumulate with it
        __shared__  ExtremeValues s_extreme;

        int startIdx = threadIdx.x * numCols;
        int endIdx  = startIdx + numCols; // exclusive
        ExtremeValues out;
        out.max = d_data[startIdx];
        out.min = out.max;
        float temp; // with this variable we just need to read the global memory once for each iteration instead of twice
        for ( int i = startIdx + 1; i < endIdx; ++i){
                temp = d_data[i];
                if (temp < out.min)
                       out.min = temp;
               if (temp > out.max)
                      out.max = temp; 
        }
        //s_extreme[threadIdx.x].max = out.max;
        //s_extreme[threadIdx.x].min = out.min;

        //atomicMin((float*)&(s_extreme.min), out.min);
        //atomicMax(&(s_extreme.max), out.max);
    // since both the atomicMin and atomicMax supports only the int, we should find someway else to solve the problem
       __syncthreads();
    if (threadIdx.x ==0){
        s_extreme.max = out.max;
        s_extreme.min = out.min;
    }
    __syncthreads();
    for (int i = 1; i < numRows; ++i){
        __syncthreads();
        if ( threadIdx.x == i){
            if (out.min < s_extreme.min)
                s_extreme.min = out.min;
            if (out.max > s_extreme.max)
                s_extreme.max = out.max;
        }
    }
        __syncthreads();

        if (threadIdx.x == 1){
                d_extreme->max = s_extreme.max;
                d_extreme->min = s_extreme.min;
        }

}

// take each row as a thread
__global__
void generateHisto(const float* d_lum, 
                   const int lumMin, 
                   const float lumRange, 
                   const int numBins, 
                   const int numRows,
                   const int numCols,
                   int *d_histo){

        // needed to be initialized, and the size of this shared memory is numRows * numBins
        extern int __shared__ s_histo[];
        for (int i = threadIdx.x; i < numRows * numBins; i += numRows)
        {s_histo[i] = 0;
         d_histo[i] = 0;
        }
        // note above is the general way to initialize shared or global memory

        int startIdx = threadIdx.x * numCols + 1;
        int endIdx  = startIdx + numCols; // exclusive
        int binStartIdx = threadIdx.x * numBins; // inclusive

        int binIdx = 0;
    
        for (int i = startIdx; i < endIdx; ++i){
                binIdx = binStartIdx + (d_lum[i] - lumMin) / lumRange * numBins;
                ++s_histo[binIdx];
        } 

        __syncthreads();

        // now starting merge the local bins
        // we have numRows threads and numBins Bins to deal with, that's numRows/numBins per Bin. but for simplicity, we use 2 threads per bin here
        if ( numRows >= 2 * numBins){ // in this case we have enough threads to run all the threads in parallel, and we use 2 threads for each bin
                // application of 2 threads reduce
                       if ( threadIdx.x < 2 * numBins){
                               binIdx = threadIdx.x /2;  // so frist two threads for the first bin, then the following two threads for the second bin, etc.
                               startIdx = ( binIdx + (threadIdx.x % 2) * numRows/2) * numBins;
                               endIdx =  startIdx + ((threadIdx.x + 1) % 2) * ( numRows/2 ) * numBins + (threadIdx.x % 2) * (numRows) * numBins;
                               int out = 0;
                               for ( int i = startIdx; i < endIdx; i += numBins)
                                      out += s_histo[i];
                               __syncthreads();

                               atomicAdd( &d_histo[binIdx], out);
                       } 
        }

        // because for this case we don't have enough threads to reduce all the bins in parallel, then it makes no sense for some threads to do it in parallel then waiting for the solo threads. so we just merge the following two cases together
        //else if (numRows >= numBins){ 

        //}
        else { // now less threads than bins
                float out = 0;
                for (int iBins = threadIdx.x; iBins < numBins; iBins += numRows){
                        for (int i = 0; i < numRows; ++i){
                                out += s_histo[iBins + i * numBins];
                        }
                        // one thread for one bin, no need to use atomicAdd
                        d_histo[iBins] = out;
                }

        }
}

// frist we just assume that the numBins is a number of two to some power

__global__
void bellochScan(const int * d_histo,
                 const int numBins,
                 unsigned int * d_cdf){
        // first we need to copy the data to the shared memory
        extern int __shared__ s_cdf[];
        // numBins/2 threads will be enough
        for ( size_t i = threadIdx.x; i < numBins; i += numBins/2){
                s_cdf[i] = d_histo[i];
        }

                __syncthreads(); // make sure the copying is finished
        // now starting the scan
        size_t startIdx = 0; // this is the starting index of the first thread which will be in used in this round
        size_t myIdx = 0;  // the index of the left element of the pair
        size_t parterIdx = 0; // the index of the right element of the pair
        size_t step_size = 0;
        for (step_size = 1; step_size < numBins; step_size <<= 1){
                __syncthreads(); // make sure all the threads have done the first round then move on to the next

                startIdx = step_size - 1;
                myIdx = startIdx + threadIdx.x * 2 * step_size; // which means the smaller the threadIdx.x is, the more often the thread will be used, and thread 0 will be at full run
                if ( myIdx < numBins - 1){  // when myIdx locates at the last element, we don't count that one
                        parterIdx = myIdx + step_size;
                        if (parterIdx >= numBins)
                                parterIdx = numBins - 1;
                        s_cdf[parterIdx] = s_cdf[parterIdx] + s_cdf[myIdx];
                }
        }
        // the up-sweep finishes here, now the following is down-sweep
        if ( threadIdx.x == 0)
                s_cdf[numBins - 1] = 0;
        __syncthreads();

        int temp;
        // the code is roughly the same as the above one
        for (step_size >>= 1; step_size >= 1; step_size >>= 1){
                __syncthreads();

                startIdx = step_size - 1;
                myIdx = startIdx + threadIdx.x * 2 * step_size;

                 if ( myIdx < numBins - 1){  // when myIdx locates at the last element, we don't count that one
                        parterIdx = myIdx + step_size;
                        if (parterIdx >= numBins)
                                parterIdx = numBins - 1;
                        temp  = s_cdf[parterIdx];
                        s_cdf[parterIdx] = temp + s_cdf[myIdx];
                        s_cdf[myIdx] = temp;
                }
       } 
        
        __syncthreads();

        // now copy back the result to the global memory
        for ( size_t i = threadIdx.x; i < numBins; i += numBins/2)
                d_cdf[i] = s_cdf[i];

}
void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

        //1) caculating the min and max
        ExtremeValues *d_extreme;
        checkCudaErrors(cudaMalloc(&d_extreme, sizeof(ExtremeValues)));
        // here a row is a block
        findExtremeValues<<<numRows, numCols>>>( d_logLuminance, numCols, numRows, d_extreme);
        min_logLum = d_extreme->min;
        max_logLum = d_extreme->max;
        
        //2) finding the range
        float range = max_logLum - min_logLum;

        //3) generating the histogram
        // allocating global memory for the histogram
        int * d_histo;
        checkCudaErrors(cudaMalloc(&d_histo, sizeof(int) * numBins));

        generateHisto<<< 1, numRows, numRows * numBins * sizeof(int) >>>( d_logLuminance, min_logLum, range, numBins, numRows, numCols, d_histo); 

        //4) the exclusive scan for cdf
        // we try to implement the Blelloch Scan here
        // numBins/2 threads are used, and a shared memory of numBins in size are allocated
        
        bellochScan<<< 1, numBins/2, numBins * sizeof(int)>>>(d_histo, numBins, d_cdf); 

}