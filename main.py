from canny_edge_detector import CannyEdgeDetector

def main():
    canny_edge_detector = CannyEdgeDetector('test_images/swan.jpg')    
    canny_edge_detector.save("edges.png")


if __name__ == '__main__':
    main()
