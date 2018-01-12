import glob
import numpy as np
import cv2
from collections import deque

class Line():
    def __init__(self, left=True, n_frames=10, debug=False):
        self.left = left # Left or right line?
        self.detected = False # was the line detected in the last iteration?
        self.recent_fit = deque(maxlen=n_frames) # circular buffer of recent frames polynomial fit
        self.recent_pts = deque(maxlen=n_frames) # circular buffer of recent frames points
        self.confidence = 0.0 # Confidence in current frame fit
        self.avg_fit = np.empty(3) # weighted average of recent_fit entries
        self.num_reset = 0 # number of times reset called to switch back to sliding window
        self.curve_radius = 0.0 # radius of curvature
        self.camera_distance = 0.0 # Distance from camera
        self.debug = debug # Save frames for debugging later
        self.color_line = None # Debug image showing colored line pixels with previous frames included
        self.window_width = 50 # Convoluation window width
        self.window_height = 48 # Convoluation window height
        self.margin = 100 # Search margin around window
        self.min_sum = 50 # minimum sum of pixels in search area to consider a new window
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        self.low_confidence_thresh = 0.01

    def get_confidence(self, img, ypts, xpts):
        xvar = max(np.var(xpts), 1.e-9)
        yprob = len(ypts)/img.shape[0]
        return yprob/xvar

    def get_fit_points(self, img, fit):
        fit_y = np.linspace(0, img.shape[0]-1, num=img.shape[0])
        fit_x = fit[0]*fit_y**2 + fit[1]*fit_y + fit[2]
        return fit_y, fit_x

    def fit_polynomial(self, img, mask):
        line_image = np.bitwise_and(img, mask)
        nonzero = line_image.nonzero()
        xpts = nonzero[1]
        ypts = nonzero[0]
        if xpts.size == 0:
            return None, 0.0
        
        self.recent_pts.append((ypts, xpts))
        combined_y = np.array([], dtype=np.int_)
        combined_x = np.array([], dtype=np.int_)
        for pts in self.recent_pts:
            combined_y, combined_x = np.concatenate((combined_y, pts[0])), \
                                     np.concatenate((combined_x, pts[1]))
         
        fit = np.polyfit(combined_y, combined_x, 2)
        confidence = self.get_confidence(img, combined_y, combined_x)
        
        if self.debug == True:
            self.color_line = np.dstack((np.zeros_like(mask), mask, np.zeros_like(mask)))*100
            self.color_line[combined_y, combined_x, :] = (0, 0, 255)
            self.color_line[ypts, xpts, :] = (255, 0, 0)
            fit_y, fit_x = self.get_fit_points(img, fit)
            line_points = np.array([np.transpose(np.vstack([fit_x, fit_y]))], dtype=np.int32)
            cv2.polylines(self.color_line, [line_points], False, color=(255,255,0), thickness=2)

        return fit, confidence

    def save_new_fit(self, cur_fit, confidence):
        self.confidence = confidence
        self.recent_fit.append(cur_fit)
        n = len(self.recent_fit)
        self.avg_fit = np.average(self.recent_fit, axis=0, weights=range(1,n+1))
        
    def reset(self):
        self.num_reset += 1
        self.detected = False

    # Find window center convolution in area bounded by box (x1, y1), (x2, y2)
    def find_window_center(self, img, y1, y2, x1, x2):
        window = np.ones(self.window_width)
        xsum = np.sum(img[y1:y2,x1:x2], axis=0)
    
        # If not enough inputs in search window, return error
        if np.sum(xsum) < self.min_sum:
            return -1
    
        # NOTE: convolution is maximum at the end of the window, so position of
        # window center is position of max convolution - window_width/2
        return np.argmax(np.convolve(window, xsum)) + x1 - self.window_width/2

    # Sliding Window Search - returns mask image for selecting left and right lane pixels
    # Below function implements sliding window convolution approach to find poteintial
    # candiadate lane line pixels. Following are hightlights for the below alogirthm:
    # 1. Obtain an initial estimate for left and right lane center by looking into bottom
    #    quarter of the image using convolution of small window of 80x50 pixels of ones over
    #    this vertical slice. Left line is searched only in first half (horizontally) of the
    #    image while right lane line is searched in second half.
    #    - If there are not enough pixels in bottom quarter, we expand this vertical slice to
    #      bottom half, 3 quarters and finally to full image. Even after this if we fail,
    #      just give up on this frame.
    # 2. Next we search for line pixels on more granular vertical slices and repeat following
    #    setps:
    #    - search for pixels around line center obtained in previous step with a horizonal
    #      margin of 100 around the center. Use convolution again to find pixels with highest
    #      density to find new values for line center.
    #   - create a window mask around this line center to select only the pixels within this
    #     window from the current vertical slice. width of the window at each vertical slice is
    #     2*margin =  200 pixels
    # 3. At this point, we should have a image mask which can be used to select only the pixels
    #    that represent this line.
    def get_sliding_window_mask(self, img):
        mask = np.zeros_like(img)

        # Lane lines are more profounds near the camera, get line center
        # with convolution on bottom portion of the image. We will use
        # these centers as starting point for search in each vertical level
        center = -1
        x_min = 0
        x_max = img.shape[1]//2
        if self.left == False:
            x_min = x_max
            x_max = img.shape[1]

        # Start searching for line pixels with bottom quarter of the image, if not enough
        # pixels in last quarter, search in lower half and so on..
        for i in range(3, -1, -1):
            center = self.find_window_center(img, y1=img.shape[0]*i//4, y2=img.shape[0], x1=x_min, x2=x_max)
            if center != -1:
                break
        if center == -1:
            raise ValueError('Line could not be detected')
     
        # Find windows at each vertical level
        for i in range(0, img.shape[0]//self.window_height):
            y_max = int(img.shape[0] - i * self.window_height)
            y_min = int(y_max - self.window_height)
        
            # Searching at each level is narrowed down with previous centroids +- margin,
            # if new window center could not be found, reuse previous center
            if self.left == True:
                x_min = int(max(center-self.margin, 0))
                x_max = int(min(center+self.margin, img.shape[1]//2))
            else:
                x_min = int(max(center-self.margin, img.shape[1]//2))
                x_max = int(min(center+self.margin, img.shape[1]))

            ret = self.find_window_center(img,  y1=y_min, y2=y_max, x1=x_min, x2=x_max)
            if ret != -1:
                center = ret
        
            # Update lane masks with new window centers
            mask[y_min:y_max, int(max(center-self.margin, 0)):int(min(center+self.margin,mask.shape[1]))] = 1
        
        return mask

    # Returns lane window mask from polyfit
    def get_polynomial_mask(self, warped_image, fit):
        mask = np.zeros_like(warped_image)
        yrange = np.linspace(0, warped_image.shape[0]-1, num=warped_image.shape[0])
        fitx = np.uint32(fit[0]*yrange**2 + fit[1]*yrange + fit[2])
        for i in range(warped_image.shape[0]):
            mask[i,fitx[i]-self.margin:fitx[i]+self.margin] = 1
        return mask

    # Get Curvature and distance from center
    def get_curvature_radius(self, img, xpts):
        yrange = np.linspace(0, img.shape[0]-1, num=img.shape[0])

        # convert input to meters
        yrange_m = yrange * self.ym_per_pix
        xpts_m = xpts * self.xm_per_pix
    
        fit_m = np.polyfit(yrange_m, xpts_m, 2)
        ymax_m = yrange_m[-1]
    
        radius = (1 + (2*fit_m[0]*ymax_m + fit_m[1])**2)**1.5/np.absolute(2*fit_m[0])
        return radius

    # Get distance of center of vehile from line
    def get_line_distance(self, img, xpts):
        return (xpts[-1] - img.shape[1]/2)*self.xm_per_pix

    def detect(self, warped_img):
        # Get mask to search for line
        mask = None
        while True:
            if self.detected == False:
                # First time or after reset from previous faulty frame,
                # use sliding window to find line
                try:
                    mask = self.get_sliding_window_mask(warped_img)
                except ValueError as e:
                    return
            else:
                mask = self.get_polynomial_mask(warped_img, self.avg_fit)

            cur_fit, confidence = self.fit_polynomial(warped_img, mask)
            if confidence < self.low_confidence_thresh:
                if self.detected == False:
                    # Sliding window search yields low confidence results,
                    # just give on on this frame
                    return
                
                # Polynomial from previous frames is not working with
                # current frame, retry with sliding window search again
                self.reset()
                continue
            
            self.save_new_fit(cur_fit, confidence)
            self.detected = True
            ypts, xpts = self.get_fit_points(warped_img, self.avg_fit)
            self.curve_radius = self.get_curvature_radius(warped_img, xpts)
            self.camera_distance = self.get_line_distance(warped_img, xpts)
            break


# Below is our LaneDetector class implementation.
class LaneDetector():
    def __init__(self, n_frames=10, debug = False):
        self.left_line = Line(left=True, n_frames=n_frames, debug=True) # Left line
        self.right_line = Line(left=False, n_frames=n_frames, debug=True) # Right line
        self.left_fit = None # Left fit for last frame
        self.right_fit = None # Right fit for last frame
        self.avg_width = 0.0 # Average width of the lane
        self.input_img = None # input image
        self.warped_img = None # warped image
        self.Minv = None # inverted perspective trnasform matrix
        self.debug = debug
        
        # Stats
        self.num_hits = 0 # number of frames with lane detected
        self.num_miss = 0 # number of frames with no lane detected

    def perspective_transform(self, img):
        xsize = img.shape[1]
        ysize = img.shape[0]
        src_pts = np.float32([[560, 470],
                              [xsize - 560, 470],
                              [xsize - 170, ysize],
                              [170, ysize]])
    
        offset = 200
        dst_pts = np.float32([[offset, 0],
                              [xsize - offset, 0],
                              [xsize - offset, ysize],
                              [offset, ysize]])
    
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # Save inverse transform matrix to later come back to real world space
        Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        warped = cv2.warpPerspective(img, M, (xsize, ysize))
        return warped, Minv

    def region_of_interest(self, img, vertices, outer=True):
        """
        Applies an image mask.
    
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        if outer == True:
            ignore_color = 255
            mask = np.zeros_like(img)
        else:
            ignore_color = 0
            mask = np.ones_like(img)
    
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (ignore_color,) * channel_count
        else:
            ignore_mask_color = ignore_color
        
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
    
        #returning the image only where mask pixels are nonzero
        return cv2.bitwise_and(img, mask)

    # Return sobel value scaled to 0-255
    def scale_image(self, image):
        return np.uint8(255*image/np.max(image))

    # Returns binary image with values filtered by threhold range
    def get_binary_threshold(self, input_img, thresh):
        binary_output = np.zeros_like(input_img)
        binary_output[(input_img >= thresh[0]) & (input_img <= thresh[1])] = 1
        return binary_output

    # Returns gradient thresholded image from single channel input image 'img'
    def get_gradient_threshold(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        HLS_l_channel = hls[:,:,1]
        HLS_s_channel = hls[:,:,2]
    
        # Get image gradients with Sobel Operator
        sobelx = cv2.Sobel(HLS_s_channel, cv2.CV_64F, 1, 0, ksize=9)
        sobely = cv2.Sobel(HLS_s_channel, cv2.CV_64F, 0, 1, ksize=9)
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        abs_sobelxy = np.sqrt(abs_sobelx**2 + abs_sobely**2)
    
        # Scale gradients
        scaled_sobelx = self.scale_image(abs_sobelx)
        scaled_sobelxy = self.scale_image(abs_sobelxy)
    
        # Direction of gradient
        grad_direction = np.arctan2(abs_sobely, abs_sobelx)
    
        # Get thresholded binary images
        grad_x_thresh = self.get_binary_threshold(scaled_sobelx, thresh=(10, 200))
        grad_xy_thresh = self.get_binary_threshold(scaled_sobelxy, thresh=(100, 255))
        grad_dir_thresh = self.get_binary_threshold(scaled_sobelxy, thresh=(np.pi/6, np.pi/2))
    
        # Filter out dark points
        HLS_l_threshold = self.get_binary_threshold(HLS_l_channel, thresh=(100, 255))
    
        # Create a combined binary image
        combined = np.zeros_like(grad_dir_thresh)
        combined[(HLS_l_threshold == 1) & \
                 ((grad_x_thresh == 1) | \
                  ((grad_dir_thresh == 1) & (grad_xy_thresh == 1)))] = 1
        combined = cv2.GaussianBlur(combined, (9,9), 0, 0)
        return combined

    # Select pixels from RGB image between rgb_min and rgb_max range
    def get_rgb_color_threshold(self, img, rgb_min, rgb_max):
        rgb_mask = cv2.inRange(img, rgb_min, rgb_max)
        color_masked = cv2.bitwise_and(img, img, mask=rgb_mask)
        gray_masked = cv2.cvtColor(color_masked, cv2.COLOR_RGB2GRAY)
        return self.get_binary_threshold(gray_masked, thresh=(20, 255))
    
    def get_color_threshold(self, img):
        # Pick up white pixels from RBG image
        min_white = np.array([100, 100, 200], dtype=np.uint8)
        max_white = np.array([255, 255, 255], dtype=np.uint8)
        rgb_w_thresh = self.get_rgb_color_threshold(img, min_white, max_white)
    
        # Pick up yellow line pixels from RGB image
        min_yellow = np.array([225, 180, 0], dtype=np.uint8)
        max_yellow = np.array([255, 255, 170], dtype=np.uint8)
        rgb_y_thresh = self.get_rgb_color_threshold(img, min_yellow, max_yellow)

        # As seen Above, Lab color space b channel can also isolate yellow line
        # in contrast to background lane. Additionally threshold based on Light
        # channel to filter out dark patches of shades
        Lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        Lab_L_channel = Lab[:,:,0]
        Lab_b_channel = Lab[:,:,2]
        Lab_L_thresh = self.get_binary_threshold(Lab_L_channel, thresh=(100, 255))
        Lab_b_thresh = self.get_binary_threshold(Lab_b_channel, thresh=(175, 255))
    
        combined = np.zeros_like(Lab_b_thresh)
        combined[((rgb_w_thresh == 1) | (rgb_y_thresh == 1) | (Lab_b_thresh == 1)) & \
                 (Lab_L_thresh == 1)] = 1
        return combined

    def get_color_gradient_threshold(self, img):
        # Get color and gradient thresholds
        gradient_threshold = self.get_gradient_threshold(img)
        color_threshold = self.get_color_threshold(img)
    
        # Combine thresholds
        combined_threshold = np.zeros_like(color_threshold)
        combined_threshold[(gradient_threshold == 1) | (color_threshold == 1)] = 1
        return combined_threshold
 
    def get_warped_image(self, img):
        warped, Minv = self.perspective_transform(img)
        thresh_img = self.get_color_gradient_threshold(warped)

        # Mask off unwanted region of image
        ysize, xsize = thresh_img.shape
        masked = np.copy(thresh_img)
        for offset, outer in zip((50, 500), (True, False)):
            mask_outer = np.int32([[offset, 0],
                                   [xsize - offset, 0],
                                   [xsize - offset, ysize],
                                   [offset, ysize]])
            masked = self.region_of_interest(masked, [mask_outer], outer)
        return masked, Minv
  
    def is_width_ok(self, width):
        if self.avg_width == 0.0:
            return True
        if (width < self.avg_width * 0.8) or (width > self.avg_width * 1.2):
            return False
        return True
    
    def is_curve_radius_ok(self):
        if (self.left_line.curve_radius < self.right_line.curve_radius * 0.8) or \
           (self.left_line.curve_radius > self.right_line.curve_radius * 1.2):
            return False
        return True
    
    def get_curve_radius(self):
        return (self.left_line.curve_radius + self.right_line.curve_radius)/2
    
    def get_center_distance(self):
        return (self.left_line.camera_distance + self.right_line.camera_distance)/2
    
    def get_lane_fit(self):
        left_fit, right_fit = None, None
        for retry_cnt in range(2):
            self.left_line.detect(self.warped_img)
            self.right_line.detect(self.warped_img)

            # We have good fit for both left and right lines, average them out
            self.num_hits += 1
            left_fit = self.left_line.avg_fit
            right_fit = self.right_line.avg_fit

            # calculate lane width
            y_max = (self.warped_img.shape[0] - 1)
            top_width = (right_fit[2] - left_fit[2])
            bottom_width = (right_fit[0] - left_fit[0])*y_max**2 + \
                           (right_fit[1] - left_fit[1])*y_max + top_width
            # If lanes are not aligned or not apart by average width or curve radius are not similar,
            # reset lane detection and retry with sliding windows
            if retry_cnt == 0:
                if not self.is_width_ok(top_width) or not self.is_width_ok(bottom_width) or \
                   not self.is_curve_radius_ok():
                    self.left_line.reset()
                    self.right_line.reset()
                    continue

            if self.avg_width == 0.0:
                self.avg_width = bottom_width
            else:
                self.avg_width = 0.6 * self.avg_width + 0.4 * bottom_width

            break

        if left_fit is not None:
            self.left_fit = left_fit
        if right_fit is not None:
            self.right_fit = right_fit
 
    def add_metrics(self, img):
        curve_radius = self.get_curve_radius()
        curvature_text = "Radius of Curvature = {}(m)".format(int(curve_radius))
        
        center_distance = self.get_center_distance()
        center_text = None
        if center_distance >= 0:
            center_text = "Vehicle is {:.2f} m left of center".format(center_distance)
        else:
            center_text = "Vehicle is {:.2f} m right of center".format(-center_distance)
        width_text = "Average Width: {:.2f}".format(self.avg_width)
        
        cv2.putText(img, curvature_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, color=(255,255,255), thickness=2)
        cv2.putText(img, center_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, color=(255,255,255), thickness=2)
        cv2.putText(img, width_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, color=(255,255,255), thickness=2)
        
        return img
    
    def get_debug_data(self):
        assert(self.debug == True)
        left_line = self.left_line.color_line
        right_line = self.right_line.color_line
        color_lines = cv2.addWeighted(left_line, 1, right_line, 1, 0)
        return color_lines
    
    # Returns unwaped line polynomial for left or right line with some margin around
    def get_unwarped_line_fit(self, left=True, margin=10):
        if left == True:
            fit = self.left_fit
            margin = -margin
        else:
            fit = self.right_fit
        
        if fit is None:
            return None
        
        # Fit last detected lane's polynomial
        yrange = np.arange(0, self.warped_img.shape[0]).astype(np.int32)
        fitx = np.int32(fit[0]*yrange**2 + fit[1]*yrange + fit[2] + margin)
        
        # create a mask for the fitted points
        mask = np.zeros_like(self.warped_img)
        mask[yrange, fitx] = 255
        
        # Unwarp mask and fit polynomial again in unwarped space
        unwarped_mask = cv2.warpPerspective(mask, self.Minv, self.input_img.shape[1::-1])
        nonzero = unwarped_mask.nonzero()
        unwarped_fit = np.polyfit(nonzero[0], nonzero[1], 2)
        
        return unwarped_fit
    
    def get_marked_lane(self):
        self.get_lane_fit()

        # If there is no center fit detected, return the original image
        if self.left_fit is None:
            assert(self.right_fit is None)
            self.num_miss += 1
            return self.input_img
        
        yrange = np.linspace(0, self.warped_img.shape[0]-1, num=self.warped_img.shape[0])
        left_fitx = self.left_fit[0]*yrange**2 + self.left_fit[1]*yrange + self.left_fit[2]
        right_fitx = self.right_fit[0]*yrange**2 + self.right_fit[1]*yrange + self.right_fit[2]

        left_points = np.array([np.transpose(np.vstack([left_fitx, yrange]))], dtype=np.int_)
        right_points = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yrange])))], dtype=np.int_)
        
        lane_poly = np.int_(np.hstack((left_points, right_points)))
        lane_mask = np.dstack((np.zeros_like(self.warped_img),)*3).astype(np.uint8)
        cv2.fillPoly(lane_mask, [lane_poly], (0,255,0))
        cv2.polylines(lane_mask, [left_points], False, (255, 255, 0), thickness=20)
        cv2.polylines(lane_mask, [right_points], False, (255, 255, 0), thickness=20)
    
        unwarped_lane_mask = cv2.warpPerspective(lane_mask, self.Minv, self.input_img.shape[1::-1])

        return unwarped_lane_mask

    def process_frame(self, img):
        self.input_img = img
        self.warped_img, self.Minv = self.get_warped_image(self.input_img)
        return self.get_marked_lane()
    
    def print_stats(self):
        print("Num Hit:        {}".format(self.num_hits))
        print("Num Miss:       {}".format(self.num_miss))
        print("Left reset:     {}".format(self.left_line.num_reset))
        print("Right reset:    {}".format(self.left_line.num_reset))
