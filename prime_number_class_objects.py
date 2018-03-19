class calculator:

	def __init__(self,a):
	
		self.num1=a
		#self.num2=b

	#def addition(self):
		
	#	return self.num1+self.num2
	#def sub(self):

	#	return self.num1-self.num2

	def prime(self):
		for i in range(2,self.num1-1):
			if self.num1%i==0:
				print "not prime"
				break
		else:
			print "prime"
			
				


if __name__ == "__main__":

	c=calculator(9)
	#p1=c.addition()
	#p2=c.sub()
	c.prime()
	
