# canny-edge-detector
The goal of this project is to create a canny edge detector from scratch. The only libraries used are numpy and Pillow. The only usage of Pillow is for easier image I/O.

To run this project, simply clone and run `python3 main.py`. To input your own files, change the path in `main.py`. Then, if you want to change the parameters, you can do so in the `CannyEdgeDetector` variable in `main.py`.

See some sample outputs below.

Image 1 ([Source](https://pixabay.com/photos/swan-bird-animal-white-swan-7842666/)): `test_images/swan.jpg`
![Origin swan image](test_images/swan.jpg)

Output 1:
![Edges for swan image](test_output/swan_edges.png)


Image 2 ([Source](https://en.wikipedia.org/wiki/Canny_edge_detector#/media/File:Valve_original_(1).PNG)): `test_images/steam_valve.png`
![Origin steam valve image](test_images/steam_valve.png)

Output 2:
![Edges for steam valve image](test_output/steam_valve_edges.png)
