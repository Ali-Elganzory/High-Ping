import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
import time
import scipy.sparse


class Video_Filter():
    def __init__(self,jump_dist,consistency_frames,bin_size=10,resize_val=1,same_threshold=1,pixel_consistency_method="binarized"):
        self.jump_dist = jump_dist
        self.consistency_frames = consistency_frames
        self.bin_size = bin_size
        self.resize_val = resize_val
        self.same_threshold = same_threshold
        self.current_clean_frame = None
        self.current_clean_bin_frame = None
        self.past_images = []
        self.past_binarized = []
        self.past_masks = []
        self.past_frames = []
        self.pixel_consistency_method = pixel_consistency_method
        assert pixel_consistency_method in ["binarized","RGB image"]

    def filtering_phase_1(self,new_frame,show_segmentation_results=False):
        self.past_frames.append(new_frame)
        if len(self.past_frames) < self.jump_dist:
            # Not enough frames accumulated
            return False,None

        # Enough frames accumulated, start filtering process 1
        if self.current_clean_frame is None: # First time to start filtering
            self.current_clean_frame = np.full((new_frame.shape[0],new_frame.shape[1],3),255).astype('uint8')
            self.current_clean_bin_frame = np.full((new_frame.shape[0],new_frame.shape[1]),1).astype('uint8')

        # Calculate diffs
        diffs = cv2.absdiff(new_frame,self.past_frames[-2])

        mask,rect = self.get_mask_from_diff(diffs,new_frame)
        self.past_masks.append(mask)

        if show_segmentation_results: # For testing purposes only
            to_show = new_frame*(1-mask.reshape(mask.shape[0],mask.shape[1],1))
            to_show = to_show.astype('uint8')
            start_point = (rect[0],rect[1])
            end_point = (rect[2],rect[3])
            color = (255, 0, 0)
            thickness = 2
            to_show = cv2.rectangle(to_show, start_point, end_point, color, thickness)
            cv2.imshow("inter",to_show)

        grey_new_frame = cv2.cvtColor(new_frame,cv2.COLOR_BGR2GRAY)
        _,binarized_new_frame = cv2.threshold(grey_new_frame,50,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.past_binarized.append(binarized_new_frame)
        self.past_images.append(self.remove_shadows(new_frame))

        if len(self.past_images) > self.consistency_frames:
            data_to_send = self.filtering_phase_2(new_frame)
            return True,data_to_send
        return False,None

    def filtering_phase_2(self,new_frame):
        # Make sure the mask is consistent for a number of "self.consistency_frames" number of frames
        consistent_mask = np.ones((new_frame.shape[0],new_frame.shape[1]))
        for j in range(-1,-self.consistency_frames-1,-1):
            consistent_mask = np.logical_and(consistent_mask,np.logical_not(self.past_masks[j]))
        consistent_mask = np.array(consistent_mask,dtype=int)

        # Make sure that pixel values for binarized images are constant, or pixel values for past_images are within a defined threshold "this.same_threshold" (2 different methods)
        same_pixel_value_mask = np.ones((new_frame.shape[0],new_frame.shape[1]))
        for j in range(-2,-self.consistency_frames-1,-1):
            to_add = None
            if self.pixel_consistency_method is "binarized":
                to_add = self.past_binarized[-1] == self.past_binarized[j]
            elif self.pixel_consistenct_method is "RGB image":
                to_add = np.sum(new_frame - self.past_images[j],axis=2) < self.same_threshold

            same_pixel_value_mask = np.logical_and(same_pixel_value_mask,to_add)

        # Combine pixel_value_mask and consistent_mask
        combined_mask = np.logical_and(same_pixel_value_mask,consistent_mask)
        self.current_clean_frame[combined_mask==1] = self.past_images[-1][combined_mask==1]

        # Remove shadows and binarize image
        colored_img = self.remove_shadows(self.current_clean_frame)
        grey=cv2.cvtColor(colored_img,cv2.COLOR_BGR2GRAY)
        _,binarized = cv2.threshold(grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((1,1),np.uint8)
        closed_img = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)
        new_clean_bin_frame = cv2.erode(closed_img,np.ones((2,2),np.uint8),iterations=1)

        #cv2.imshow("inter",(new_clean_bin_frame).astype('uint8'))
        data_to_send = self.encode_data(new_clean_bin_frame)
        self.current_clean_bin_frame = new_clean_bin_frame
        return data_to_send

    def encode_data(self,new_clean_bin_frame):
        ind_x,ind_y = np.where(new_clean_bin_frame!=self.current_clean_bin_frame)
        vals = new_clean_bin_frame[new_clean_bin_frame!=self.current_clean_bin_frame]/255
        vals = vals.astype('bool')
        indices = np.vstack((ind_x,ind_y)).transpose().astype('uint16')
        shape = np.array([new_clean_bin_frame.shape[0],new_clean_bin_frame.shape[1]],dtype=np.uint16)
        return [indices.tolist(),vals.tolist(),shape.tolist()]

    def custom_hull(self,data,max_y):
        min_x = np.array(np.min(data[:,0]),dtype=int)
        max_x = np.array(np.max(data[:,0]),dtype=int)
        data_x = data[:,0]
        data_y = data[:,1]
        hull_min = []
        hull_max = []
        for i in range(min_x,max_x,self.bin_size):
            mask = np.logical_and(data_x>i,data_x<i+self.bin_size)
            y_in_range = data_y[mask]
            if len(y_in_range)>0:
                x_in_range = data_x[mask]
                idx_min = np.argmin(y_in_range)
                idx_max = np.argmax(y_in_range)
                hull_min.append([x_in_range[idx_min],y_in_range[idx_min]])
                hull_max.append([x_in_range[idx_max],max_y])
        for h in range(len(hull_max)-1,0,-1):
            hull_min.append(hull_max[h])
        hull_min.append([hull_max[0][0],max_y])
        hull_min = np.array(hull_min,dtype=np.int32)
        return hull_min

    def get_mask_from_diff(self,diffs,frame,verbose=False):
        start = time.time()

        # Resize diffs and frame
        resized_diffs = cv2.resize(diffs,(diffs.shape[1]//self.resize_val,diffs.shape[0]//self.resize_val))
        resized_frame = cv2.resize(frame,(diffs.shape[1]//self.resize_val,diffs.shape[0]//self.resize_val))

        # Apply phase 1 filters and detect contours
        grey=cv2.cvtColor(resized_diffs,cv2.COLOR_BGR2GRAY)
        blur =cv2.GaussianBlur(grey,(9,9),0)
        ret,th=cv2.threshold(blur,20,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        dilated=cv2.dilate(th,np.ones((1,1),np.uint8),iterations=3)
        c,h=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(resized_frame.shape)
        for contour in c:
            cv2.fillPoly(mask,[contour],(3,3,3))

        # Apply morphological closing on detected contours
        kernel = np.ones((3,3),np.uint8)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Determine surrounding rectangle and hull
        pos_y,pos_x,_ = np.where(closing==3)
        rect = (np.min(pos_x),np.min(pos_y),np.max(pos_x),np.max(pos_y))
        data = np.vstack((pos_x,pos_y)).transpose()
        hull = self.custom_hull(data,diffs.shape[0])

        # Convert detected hull polygon to mask
        mask = np.zeros(resized_frame.shape)
        cv2.fillPoly(mask,[hull.reshape(-1,1,2)],(1,1,1))

        # Resize mask to original image size and dilate the mask
        fix_size_mask = cv2.resize(mask,(diffs.shape[1],diffs.shape[0]))[:,:,0]
        dilated=cv2.dilate(fix_size_mask,np.ones((10,10),np.uint8),iterations=3)

        # Print filtering details such as time taken, contours detecting points, and hull results
        if verbose:
            time_used = time.time()-start
            print("Filtering time: {}ms".format(time_used*1000))
            x = []
            y = []
            for h in hull:
                x.append(h[0])
                y.append(h[1])
            plt.fill(x,y, facecolor='none', edgecolor='purple', linewidth=3)
            plt.scatter(data[data_idx][:,0],data[data_idx][:,1])
            plt.show()
        return dilated,rect

    def remove_shadows(self,img):
        rgb_planes = cv2.split(img)

        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)

        result = cv2.merge(result_planes)
        result_norm = cv2.merge(result_norm_planes)
        return result_norm

class Client_Reconstructor():
    def __init__(self):
        self.current_bin_frame = None

    def receive_data(self,data,show_frame=True):
        indices = data[0]
        vals = data[1]
        img_shape = data[2]
        indices = np.array(indices)
        vals = np.array(vals)
        if indices.shape[0]>0:
            if self.current_bin_frame is None: # First data received:
                self.current_bin_frame = np.ones(img_shape)
            changes_to_frame,changes_mask = self.decode(indices,vals,img_shape)
            self.current_bin_frame[changes_mask==1] = changes_to_frame[changes_mask==1]
        if show_frame:
            cv2.imshow("inter",(self.current_bin_frame*255).astype('uint8'))
        return self.current_bin_frame
    def decode(self,indices,vals,img_shape):
        recons = scipy.sparse.coo_matrix((vals,indices.transpose()),img_shape).toarray()
        to_change = scipy.sparse.coo_matrix((np.ones(indices.shape[0]),indices.transpose()),img_shape).toarray()
        return recons.astype('uint8'),to_change

class Instructor_Encoder():
    def __init__(self,jump_dist,consistency_frames):
        self.VF = Video_Filter(jump_dist,consistency_frames)
        self.cap = None
        self.network = None
        self.images = None
        self.current_idx = None

    def start_camera_enc(self,network):
        self.cap = cv2.VideoCapture(0)
        self.network = network

    def send_data_camera(self):
        ret, frame = self.cap.read()
        if frame is not None:
            data_found,data = self.VF.filtering_phase_1(frame)
            if data_found:
                self.network.send_screen_update(data)
                
    def end_camera(self):
        self.cap.release()

    def start_file_enc(self,file_name,start_frame,end_frame,network):
        self.network = network
        vid = cv2.VideoCapture(file_name)
        images = []
        for num in range(0,end_frame):
            _,image = vid.read()
            images.append(image)
        self.images = images[start_frame:]
        self.current_idx = 0

    def send_data_file(self,network):
        img = self.images[self.current_idx]
        self.current_idx += 1
        data_found,data = self.VF.filtering_phase_1(img)
        if data_found:
            network.send_screen_update(data)
