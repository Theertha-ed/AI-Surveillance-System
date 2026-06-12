from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=30, n_init=2, max_cosine_distance=0.3)

print("DeepSORT Tracker Ready")