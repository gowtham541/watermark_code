x,y= 0,0
for i in range (0,1000):
    for j in range (0,1000):
        for i in range(3):
            for j in range(3,6):
                if x == 230:
                    break
                else:
                    print(x,y)
                    # part8x8[i,j] +=beta*dct_bsrc[x,y]
                    y+=1
                    if y == 230:
                        y=0	
                        x+=1