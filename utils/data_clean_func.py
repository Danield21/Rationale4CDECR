# The following are some functions being used to process the text captured from the LLM.
def remove_serial_num(x):
    '''Remove the serial number from the text 
    '''
    return x.split('. ')[-1]


def removeRedundancy_then_splitCandidate(x):
    ''' Remove the redundancy '\n' and split the candidate into a list
    '''
    return x.strip('\n').split('\n')


