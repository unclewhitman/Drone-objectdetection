def generate_slices(width, height, slice_size=896, overlap=0.25):
    if slice_size <= 0:
        raise ValueError("slice_size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")

    step = max(1, int(round(slice_size * (1.0 - overlap))))
    xs = list(range(0, max(width - slice_size, 0) + 1, step))
    ys = list(range(0, max(height - slice_size, 0) + 1, step))
    if not xs or xs[-1] != max(width - slice_size, 0):
        xs.append(max(width - slice_size, 0))
    if not ys or ys[-1] != max(height - slice_size, 0):
        ys.append(max(height - slice_size, 0))

    boxes = []
    for y1 in ys:
        for x1 in xs:
            x2 = min(x1 + slice_size, width)
            y2 = min(y1 + slice_size, height)
            boxes.append((x1, y1, x2, y2))
    return boxes
