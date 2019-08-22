import numpy as np
import matplotlib.pyplot as plt
from treat_data import my_size, is_an_outlierD


def my_spike_removal(x,y,n_sigma,abs_thresh,return_type=0):
    clust_size = 5
    
    size = my_size(x,y)
    cut_vec = np.zeros(size)
    
    x_new = np.empty(0)
    y_new = np.empty(0)
    x_removed = np.empty(0)
    y_removed = np.empty(0)

    disc_arr = np.empty(0)
    rms_arr = np.empty(0)
    
    #add the first data points in by hand
    x_new =  np.append(x_new,x[0:clust_size])
    y_new =  np.append(y_new,y[0:clust_size])
    cut_vec[:clust_size] = 1
    
    #loop through the bulk of the data
    for i in range(clust_size,size-clust_size):
       
        #characterize the pre-cluster
        pre_clust_x = x[(i-clust_size):(i)]
        pre_clust_y = y[(i-clust_size):(i)]
        #mean
        pre_avg = np.median(pre_clust_y)
        pre_variance = np.var(pre_clust_y)
        pre_pos = np.mean(pre_clust_x)
        #min
        min_pre_val = np.amin(pre_clust_y)
        min_pre_pos = pre_clust_x[np.argmin(pre_clust_y)]
        #max
        max_pre_val = np.amax(pre_clust_y)
        max_pre_pos = pre_clust_x[np.argmax(pre_clust_y)]
        
        #characterize the post-cluster
        post_clust_x = x[(i+1):(i+clust_size+1)]
        post_clust_y = y[(i+1):(i+clust_size+1)]
        #mean/median/rms
        post_avg = np.median(post_clust_y)
        post_variance = np.var(post_clust_y)
        post_pos = np.mean(post_clust_x)
        #min
        min_post_val = np.amin(post_clust_y)
        min_post_pos = pre_clust_x[np.argmin(post_clust_y)]
        #max
        max_post_val = np.amax(post_clust_y)
        max_post_pos = pre_clust_x[np.argmax(post_clust_y)]
        
        #set true if the the point is above or below all points in it's neighborhood
        sits_outside = above_or_below(y[i],min_pre_val,min_post_val,max_pre_val,max_post_val)
        
    

        #get the total error on the interpolation (this ignores x values)
        total_rms = np.sqrt(post_variance + pre_variance)
        rms_arr = np.append(rms_arr,total_rms)
        
        #Set the frequency threshold based on the clusters' rms (and the user input)
        thresh = n_sigma*total_rms
        thresh += abs_thresh
        

        is_outlier, y_disc = is_an_outlierD(x[i],pre_pos,post_pos,y[i],pre_avg,post_avg,thresh,True)
        disc_arr = np.append(disc_arr,y_disc)
#        print("point", i," rms=",total_rms," thesh=",thresh, "outly=",is_outlier," sitsout=",sits_outside)
#        print("minpre={}  minpost={}  maxpre={}  maxpost={}".format(min_pre_val,min_post_val,max_pre_val,max_post_val))
#        print("")

        
        #Include the point i if:
        #it does not sit outside its neighbors OR
        #if it is not an interpolation outlier
        if ((not(sits_outside)) or (not(is_outlier))):
            x_new = np.append(x_new, x[i])
            y_new = np.append(y_new,y[i])
            cut_vec[i] = 1
        #if specified in the function call, fill the vectors of all removed points
        elif (return_type==1):
            x_removed = np.append(x_removed, x[i])
            y_removed = np.append(y_removed, y[i])

    #add the last data points in by hand
    x_new = np.append(x_new,x[size - clust_size:size])
    y_new = np.append(y_new,y[size - clust_size:size])
    cut_vec[size - clust_size:size] = 1
    
    if (return_type==0):
        return x_new,y_new
    
    elif (return_type==1):
        return x_new,y_new, x_removed, y_removed
    elif (return_type==2):
        return cut_vec
    elif (return_type==3):
        return disc_arr, rms_arr




###########################Support Functions#######################
def above_or_below(y,min1,min2,max1,max2):
    if( (y>max1) and (y>max2)):
        return True
    elif( (y<min1) and (y<min2)):
        return True
    else:
        return False



def get_derivative(x,y):
    size = my_size(x,y)
    pre_v = np.empty(0)
    post_v = np.empty(0)
    for i in range(size):
        
        #special case for the first point
        if(i==0):
            num = y[i+1] - y[i]
            den = x[i+1] - x[i]
            if (abs(den)>0):
                vel = num/den
            else:
                vel = 0
            pre_v = np.append(pre_v,vel)
            post_v = np.append(post_v,vel)
        
        
        #special case for the last point
        elif(i==(size-1)):
            num = y[i] - y[i-1]
            den = x[i] - x[i-1]
            if (abs(den)>0):
                vel = num/den
            else:
                vel = 0
            pre_v = np.append(pre_v,vel)
            post_v = np.append(post_v,vel)
        
        
        
        #for non-edge points, the velocity is calculated seperately for both sides
        else:
            
            #do the pre
            num = y[i] - y[i-1]
            den = x[i] - x[i-1]
            if (abs(den)>0):
                vel_pre = num/den
            else:
                vel = 0
            pre_v = np.append(pre_v,vel)
            
            #do the post
            num = y[i+1] - y[i]
            den = x[i+1] - x[i]
            if (abs(den)>0):
                vel = num/den
            else:
                vel = 0
            post_v = np.append(post_v,vel)
#        print ("pre/post= {} / {}".format(pre_v[i],post_v[i]))


    return pre_v, post_v

