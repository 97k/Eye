# Eye

Tracks the eye pupil's movement and effectively detects the blinking of eye. Blinking of eye can be easily detected when eye(facial) landmarks are available, Eye is a computer vision application which can effectively tracks the pupil and find the eye blinking based on heuristics powered by data.

## Instructions to Run

Clone the project
```sh
$ git clone https://github.com/aditya98ak/Eye.git
```

Install the dependencies, [create a virtual environment (recommended)](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/)
```sh
$ cd Eye
$ pip install -r requirements.txt
```
## Run the application

```sh
$ python main.py
```
I've added a video file in order to show how it works. As of now, this application works on closely cropped eye videos stacked side by side (horizontally). The aim of this application is to apply heuristics to find blinking of the eye in case where eye(facial) landmarks are not available or can not be generated. This is just a POC and it's performace can definately be improved!

## Final Result

![eye](https://user-images.githubusercontent.com/21143936/73135248-04d2ff00-4066-11ea-9c28-c4d369479ba7.gif)

## Screenshots

1.  Sample frame
![image](https://user-images.githubusercontent.com/21143936/73131590-c7ee1480-4033-11ea-90bb-c859d108b3be.png)
2.  Detection of pupil
![image](https://user-images.githubusercontent.com/21143936/73131548-f7e8e800-4032-11ea-868d-dfca3cfe20e3.png)
3.  Detection of blink
![image](https://user-images.githubusercontent.com/21143936/73131577-852c3c80-4033-11ea-95b3-52d68bf3fd0f.png)
4. SSIM Difference
![image](https://user-images.githubusercontent.com/21143936/73131600-18657200-4034-11ea-85f1-57726f04e3e0.png)

This approach can be helpful in cases where data is obtained from devices in healthcare domain and we may still need to track, capture blink of eye and not whole face is visible. If whole face is visible, then eye blink can be easily detected with help of (eye) facial landmarks!

## Contributing

All patches welcome!
