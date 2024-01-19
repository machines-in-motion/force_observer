# Moving Average Filter
def moving_average_filter(input_data, filter_size=1):
    '''
    moving average on 1st dimension of some array ((N,n))
    '''
    output_data = input_data.copy() 
    # Filter them with moving average
    for i in range( output_data.shape[0] ):
        n_sum = min(i, filter_size)
        # Sum up over window
        for k in range(n_sum):
            output_data[i,:] += input_data[i-k-1, :]
        #  Divide by number of samples
        output_data[i,:] = output_data[i,:] / (n_sum + 1)
    return output_data 
    