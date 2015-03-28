__kernel void hello(__global float* inputs, __global float* weights, __global float* outputs, __global int* numInputs)
{
	// Every thread has an ID. 
	int id = get_global_id(0);

	// Create local output variable to avoid gpu main memory access
	float output = 0;
	long offset = id*get_global_size(0);

	// We add one to the ID and return that.
	for (int i=0; i<1024; i++) {
		output += inputs[i]*weights[i+offset];
	}

	// add bias
	output += 0.5;

	// Actually write output to main GPU memory
	outputs[id] = output; //get_global_size(0);//output; //get_local_size(0);
}
