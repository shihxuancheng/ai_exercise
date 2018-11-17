
#%%
def bag(n,c,w,v):
	res=[[-1 for j in range(c+1)] for i in range(n+1)]
	for j in range(c+1):
		res[0][j]=0
	for i in range(1,n+1):
		for j in range(1,c+1):
			res[i][j]=res[i-1][j]
			if j>=w[i-1] and res[i][j]<res[i-1][j-w[i-1]]+v[i-1]:
				res[i][j]=res[i-1][j-w[i-1]]+v[i-1]
	return res
 
def show(n,c,w,res):
	print('Most Valueavle:',res[n][c])
	x=[False for i in range(n)]
	j=c
	for i in range(1,n+1):
		if res[i][j]>res[i-1][j]:
			x[i-1]=True
			j-=w[i-1]
	print('Selected Item:')
	for i in range(n):
		if x[i]:
			print('Number',i,end='')
	print('')
 
if __name__=='__main__':
	n=6
	c=20
	w=[10,9,4,2,1,20]
	v=[175,90,20,50,10,200]
	res=bag(n,c,w,v)
	show(n,c,w,res)
