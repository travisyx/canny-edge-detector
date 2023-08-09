from canny_edge_detector import CannyEdgeDetector


def main():
    canny_edge_detector = CannyEdgeDetector("test_images/steam_valve.png")
    canny_edge_detector.save("test_output/steam_valve_edges.png")


if __name__ == "__main__":
    main()
