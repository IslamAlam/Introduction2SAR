

COEF_VAR_DEFAULT = 0.01
CU_DEFAULT = 0.25


def assert_window_size(win_size):
    """
    Asserts invalid window size.
    Window size must be odd and bigger than 3.
    """
    assert win_size >= 3, 'ERROR: win size must be at least 3'

    if win_size % 2 == 0:
        print('It is highly recommended to user odd window sizes.'\
              'You provided %s, an even number.' % (win_size, ))


def assert_indices_in_range(width, height, xleft, xright, yup, ydown):
    """
    Asserts index out of image range.
    """
    assert xleft >= 0 and xleft <= width, \
        "index xleft:%s out of range (%s<= xleft < %s)" % (xleft, 0, width)

    assert xright >= 0 and xright <= width, \
        "index xright:%s out of range (%s<= xright < %s)" % (xright, 0, width)

    assert yup >= 0 and yup <= height, \
        "index yup:%s out of range. (%s<= yup < %s)" % (yup, 0, height)

    assert ydown >= 0 and ydown <= height, \
        "index ydown:%s out of range. (%s<= ydown < %s)" % (ydown, 0, height)


def lee_filter(img, win_size=3, cu=CU_DEFAULT):
    """
    Apply lee to a numpy matrix containing the image, with a window of
    win_size x win_size.
    """
    assert_window_size(win_size)

    # we process the entire img as float64 to avoid type overflow error
    img = np.float64(img)
    img_filtered = np.zeros_like(img)
    N, M = img.shape
    win_offset = win_size / 2

    for i in range(0, N):
        xleft = i - win_offset
        xright = i + win_offset

        if xleft < 0:
            xleft = 0
        if xright >= N:
            xright = N

        for j in range(0, M):
            yup = j - win_offset
            ydown = j + win_offset

            if yup < 0:
                yup = 0
            if ydown >= M:
                ydown = M

            assert_indices_in_range(N, M, xleft, xright, yup, ydown)

            pix_value = img[i, j]
            window = img[xleft:xright, yup:ydown]
            w_t = weighting(window, cu)
            window_mean = window.mean()
            new_pix_value = (pix_value * w_t) + (window_mean * (1.0 - w_t))

            assert new_pix_value >= 0.0, \
                    "ERROR: lee_filter(), pixel filtered can't be negative"

            img_filtered[i, j] = round(new_pix_value)

    return img_filtered


figure_size = 9
def conservative_smoothing_gray(data, filter_size):
    temp = []

    indexer = filter_size // 2

    new_image = data.copy()

    nrow, ncol = data.shape

    for i in range(nrow):

        for j in range(ncol):

            for k in range(i-indexer, i+indexer+1):

                for m in range(j-indexer, j+indexer+1):

                    if (k > -1) and (k < nrow):

                        if (m > -1) and (m < ncol):

                            temp.append(data[k,m])

            temp.remove(data[i,j])


            max_value = max(temp)

            min_value = min(temp)

            if data[i,j] > max_value:

                new_image[i,j] = max_value

            elif data[i,j] < min_value:

                new_image[i,j] = min_value

            temp =[]

    return new_image.copy()
img = img_sar[1]
new_image = conservative_smoothing_gray(img, figure_size)