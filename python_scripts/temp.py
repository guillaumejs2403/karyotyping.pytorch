import time

class Timer(object):
	def __init__(self, name=None):
		self.name = name

	def tic(self):
		self.tstart = time.time()
		
	def toc(self):
		return time.time() - self.tstart
		
	def wait(self,seconds=1):
		if(seconds>0):
			time.sleep(seconds)

def get_time(ETF):
    if ETF%3600 != ETF:
        string = '{0:.2f}'.format(ETF/3600)
        return [string,'hours']
    elif ETF%60 != ETF:
        string = '{0:.2f}'.format(ETF/60)
        return [string,'minutes']
    else:
        string ='{0:.2f}'.format(ETF)
        return [string,'seconds']


def print_message(epoch, timer, n_epochs):
    print('='*75)
    if epoch == 0:
        timer.tic()
        print('Epoch:',epoch+1,'/',n_epochs,'| Initiating timer')
    else:
        ETF = timer.toc()
        time, units = get_time(int((n_epochs - epoch)*ETF//(epoch+1)))
        print('Epoch ==>',epoch+1,'/',n_epochs,'| Estimated time:',time ,units)
    print('='*75)