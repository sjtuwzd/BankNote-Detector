# importing os module 
import os 
  
# Function to rename multiple files 
def main(): 
    i = 13
    j = 13
      
    for filename in os.listdir("data"):
        if "IND" in filename:
            dst = ""
            pos = filename.find(".")
            if filename[pos-1] == 'o':
                dst ="500_0_" + str(i) + ".jpg"
                i += 1
            else:
                dst ="500_1_" + str(j) + ".jpg"
                j += 1
            src = "data/" + filename
            dstnew = "data/" + dst
            os.rename(src, dstnew)

        # if "1000_1_" in filename:
        #     dst = filename
        #     pos = filename.find(".")
        #     i = int(dst[7:pos])
        #     i = i-1
        #     dstnew = "data/1000_1_" + str(i) + ".jpg"
        #     src = "data/" + filename
        #     os.rename(src, dstnew)

  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 