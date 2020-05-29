# Python/OpenCV Speed Detector #

This is a program that uses OpenCV to calculate cars' speeds from a traffic cam livestream.

### How it works ###

This demo gif will be referenced multiple times in the explanation, so I'll just leave it here.

![Example](./demo.gif)

#### Cropping ####

The first thing my program does to the video is crop out any unnecessary areas. In the gif below, the black box is blocking out a part of the screen that has motion but shouldn't be part of our detection. These cropped regions can be manually selected at runtime (click and drag on the "Source Image" window) and are saved in `settings.json` (when 's' key is pressed). Saved regions are cropped out on startup.

#### Vehicle Detection ####

Now that the unwanted areas are removed, we can use computer vision to isolate the vehicles (after all, that's what we really care about!). 

I use KNN background subtraction and morphology to isolate the vehicles and detect their contours. I'm not going to explain too much since these are default OpenCV functions, but you can see how I use them in the first part of `process_frame()` and `filter_mask()` in `main.py`.

#### Vehicle Tracking ####

To find a car's speed, we need to know how its moving from frame to frame. We can already detect cars on any given frame, but we need a kind of permanence to detect as the move in the video. This is a rather long process, but in general we compare the current detections to the previous detections, and based on manually set parameters, we determine whether or not the new detections are valid movements. The StackOverflow post in the credits goes does a much better job of explaining this.

#### Speed Calculation ####

This program has two methods of detecting speed: *distance mode* and *average mode*.

*Distance mode* will takes in a preset "distance" value (how long the road in the video is). The program uses this value and the vehicle's time on screen to calculate its speed.

*Average mode* samples a certain number of vehicles to find there average speed on screen (in pixels). Subsequent cars are compared to the average, and their speeds are reported as percent differences from the average. This mode is useful when you don't know the distance of the road in the video, so it can be applied to almost any road. The demo gif is calculating speed in average mode.

It's important to note that speed is calculated once a vehicle passes the light blue line (again, see the demo gif). The position and angle (i.e. horizontal/vertical) can be customized for different roads/video sources.

#### Settings ####

I designed this program so that it could work on virtually any video source.

`settings.json` stores settings for each individual video source (I call them "roads" in my program). A few examples include the positon of the detection line, the url of the video source, and the cropped out regions. Of course, you can look at `settings.json` to see how I actually store these values.

### Improvements ###

While my program works decently well, there are some things I'd like to work on, namely:
* A manual livestream loader to reduce lag/random jumps (it would need to be manual because it needs to access [.m3u8 metadata](https://tools.ietf.org/html/rfc8216#section-4.3))
* Lane detection so that this program could work on multi-lane roads

### Credits ###

This [incredible StackOverflow post](https://stackoverflow.com/questions/36254452/counting-cars-opencv-python-issue/36274515#36274515), which largely inspired me to redo this project in the first place.

The [livestreams](http://dot.ca.gov/d3/cameras.html) are provided by the California Department of Transportation