#include <stdio.h>
#include <stdlib.h>
 
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#define MAX_SOURCE_SIZE (0x100000)
 
void checkErr(int err) {
    if (err != 0) {
        printf("\nError: return code %i\n\n", err);
    }
}

int main() {
    int size = 1024;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem memobj = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret, err;

    /* This is were you set the amount of the threads per a kernal */
    size_t work_units_per_kernel;
    
     
    float string[size];
     
    FILE *fp;
    char fileName[] = "./hello.cl";
    char *source_str;
    size_t source_size;
     
    /* Load the source code containing the kernel*/
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
     
    /* Get Platform and Device Info */
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
     
    /* Create OpenCL context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
     
    /* Create Command Queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
     
    /* Create Memory Buffer */
    memobj = clCreateBuffer(context, CL_MEM_READ_WRITE,size * sizeof(float), NULL, &ret);
     
    /* Create Kernel Program from the source */
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
    (const size_t *)&source_size, &ret);
     
    /* Build Kernel Program */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    checkErr(ret);
    

    /* Create OpenCL Kernel */
    kernel = clCreateKernel(program, "hello", &ret);
    checkErr(ret);



    /* Set OpenCL Kernel Parameters */

    float inputs[size];
    float weights[size*size];
    float outputs[size];
    int numInputs = size;


    printf("\ninputs:\n");
    for (int i=0; i<size; i++) {
        inputs[i] = (rand()*1.0)/RAND_MAX;
    }
    for (int i=0; i<10; i++) {
        printf("(%i,%f)\n", i, inputs[i]);
    }

    printf("\n\nweights:\n");
    for (int i=0; i<(size*size); i++) {
        weights[i] = (rand()*0.01)/RAND_MAX;
    }
    for (int i=0; i<10; i++) {
        printf("(%i,%f)\n", i, weights[i]);
    }
    printf("\n");

    cl_mem dev_inputs = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*size, inputs, &err);
    checkErr(err);
    cl_mem dev_weights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*size*size, weights, &err);
    checkErr(err);
    cl_mem dev_outputs = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*size, outputs, &err);
    checkErr(err);
    cl_mem dev_numInputs = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), &numInputs, &err);
    checkErr(err);


    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&dev_inputs);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&dev_weights);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&dev_outputs);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&dev_numInputs);
     
    /* Execute OpenCL Kernel */
    //ret = clEnqueueTask(command_queue, kernel, 0, NULL,NULL);
    work_units_per_kernel = size;
    cl_uint work_dims = 1;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, work_dims, NULL, &work_units_per_kernel, NULL, 0, NULL, NULL);
    checkErr(err);

    /* Copy results from the memory buffer */
    ret = clEnqueueReadBuffer(command_queue, dev_outputs, CL_TRUE, 0, size * sizeof(float), string, 0, NULL, NULL);
    

    /* Display Result */
    //sprintf(string);
    printf("outputs:\n");
    for (int i=0; i<10; i++) {
        printf("(%i,%f)\n", i, string[i]);
    }
     
    /* Finalization */
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(memobj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    /* Free the memory we manloced. */
    free(source_str);
     
    return 0;
}
