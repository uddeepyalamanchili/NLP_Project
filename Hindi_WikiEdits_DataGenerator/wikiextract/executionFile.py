total=7823271
if total%3 :raise Exception("check the augmented file")
total_edits=total//3
train=int(total_edits*.8)*3
val = total - train 
tmp3=train+val
print(train,val)
print('tmp3 ', tmp3)