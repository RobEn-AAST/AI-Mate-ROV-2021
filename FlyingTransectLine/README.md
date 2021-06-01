# Task 2.1

Flying a transect line over a coral reef.

Flying the submarine between two blue pipes making both blue pipes are always visible in the submarine's camera and stand within a certain range. 

## Requirements

- python 3 used version v3.8.5 .
- open-cv used version v4.4 .
- numpy used version v1.19.4 .

## Installation

```bash
pip install opencv-python
pip install numpy
```

## Usage

Change the constants at the first of the python file to your needs.

```python
''' CONSTANTS '''
L_RANGE = PipeRange(0.08, 0.3) 
R_RANGE = PipeRange(0.7, 0.92) 
BLUE_PIPE_COLOR_RANGE = [[99, 173, 80], [112, 255, 174]]
PIPES_DISTANCE = [0.78, 0.62]  
CAPTURE_FROM = "vid.mp4"
''' ^^^^^^^^^ '''
```
- `L_RANGE ` and `R_RANGE ` Is the permitted range for the blue pipes within X-axis in percentage relative to the width of the full-screen, where `L_RANGE ` is the range for the left pipe and `R_RANGE ` for the right pipe.

- `BLUE_PIPE_COLOR_RANGE `The HSV color range for the blue pipes where [0]: min-hsv-range [1]:max-hsv-range.

- `PIPES_DISTANCE ` Permitted distance between both blue pipes in percentage relative to width of full-screen  where [0]:max-distance, [1]:min-distance.

- `CAPTURE_FROM` This constant can either take a __path for a video file or an integer value__ referring to an external camera. __e.g.__```CAPTURE_FROM = "vid.mp4"``` :film_strip: or ```CAPTURE_FROM = 0``` :camera: .


