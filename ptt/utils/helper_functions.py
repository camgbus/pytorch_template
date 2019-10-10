import datetime
def get_time_string(cover=False):
    '''
    Returns the current time in the format YYYY-MM-DD_HH-MM
    '''
    date = str(datetime.datetime.now()).replace(' ', '_').
        replace(':', '-').split('.')[0][:-3]
    return date