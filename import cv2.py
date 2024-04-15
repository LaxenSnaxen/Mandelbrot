import cv2
import numpy as np

coords = [] # Här sparas koordinaterna som användaren klickar på


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN: # Om vänster musknapp klickas
        coords.append([x, y])
        print(x, y)


def getComplexPlane(start, stop, num):
    # Skaffar arrays från xy arrays
    xSpace = np.linspace(start[0], stop[0], int(num[0]))
    ySpace = np.linspace(start[1], stop[1], int(num[1]))
    # Meshgrids (det mellan det riktiga och komplexa nummer, en mesh av de båda)
    real, imag = np.meshgrid(xSpace, ySpace[::-1])
    # Bygger den komplexa planan
    c = np.zeros_like(real, dtype=np.cdouble)
    c.real = real
    c.imag = imag

    return c


def f(z, c):
    return z**2 + c # beräkningen för mandelbrot


def applyNTimes(z, c, last_n, n, escape):
    # Applicerar beräkningen ett antal gånger
    for _ in range(last_n, n):
        z = f(z, c)
        m = np.abs(z) < 2
        escape[m] += 1
        if np.sum(m) == 0:
            break
    return escape


def colourImage(image, mod, cMap=cv2.COLORMAP_TWILIGHT_SHIFTED):
    # Get mandelbrot set
    mask = image == np.max(image)
    # Modulo bild
    image %= mod
    # Scale values to 8bit image and convert type
    image = (255 / (mod - 1) * image).astype(np.uint8)
    # Färgar
    image = cv2.applyColorMap(image, cMap)
    # Gör självaste mandelbrot svart
    image[mask] = [0, 0, 0]
    return image


def textOverlay(image, center, radius, maxIter, resolution):
    # Setup för text
    h = resolution[1]
    fontScale = h / 1300
    fontHeight = 30 * fontScale

    c = np.around(center, 3)
    if radius[1] < 1:
        r = [np.format_float_scientific(x, 3, trim="-") for x in radius]
    else:
        r = np.around(radius, 3)

    # Lägger till text
    image = cv2.putText(
        image,
        (
            f"center: ({c[0]}, {c[1]}i)     radius: ({r[0]}, {r[1]})    "
            + f"resolution: {resolution[0]}x{resolution[1]}    iterations {maxIter}"
        ),
        (int(0.5 * fontHeight), int(1.5 * fontHeight)),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale,
        (255, 255, 255),
        thickness=max(1, int(fontScale)),
    )
    return image


def reCenter(coords, c):
    # Ändrar vart centern bilden är, beroende på var användaren klickar
    cPoint = c[coords[-1][1], coords[-1][0]]

    center[0] = cPoint.real
    center[1] = cPoint.imag
    start = center - radius
    stop = center + radius

    return center, start, stop


def zoomIn(radius):
    # Zoomar in
    radius /= 2
    start = center - radius
    stop = center + radius

    return radius, start, stop


def zoomOut(radius):
    # Zoomar ut
    radius *= 2
    start = center - radius
    stop = center + radius

    return radius, start, stop


def maxIterUp(maxIter):
    # Ökar maxiter, vilket ger mer detalj
    lastMaxIter = maxIter
    maxIter *= 2
    return lastMaxIter, maxIter


def maxIterDown(maxIter):
    # Minskar maxiter, vilket ger mindre detalj
    lastMaxIter = 0
    maxIter //= 2
    return lastMaxIter, maxIter


def saveImage(image, start, stop):
    # Sparar bilden som en png
    xStr = f"{str(start[0]).replace('.','_')}-{str(stop[0]).replace('.','_')}"
    yStr = f"{str(start[1]).replace('.','_')}-{str(stop[1]).replace('.','_')}"

    # Sparar bilden i samma mapp som programmet
    cv2.imwrite(f"x-{xStr}-y-{yStr}-m-{maxIter}-c-{cMap}-{mod}.png", image)


def rollcMap(cMap):
    # Ändrar färg karta
    cMap += 1
    cMap %= 22
    return cMap


def resolutionUp(k, aspecRatio):
    # Ökar resolutionen
    k = int(k * 1.2)
    resolution = np.array([aspectRatio * k, k], dtype=int)
    return k, resolution


def resolutionDown(k, aspectRatio):
    # Sänker resolutionen
    k = int(k / 1.2)
    resolution = np.array([aspectRatio * k, k], dtype=int)
    return k, resolution


if __name__ == "__main__":
    k = 215
    aspectRatio = 16 / 9
    resolution = np.array([aspectRatio * k, k], dtype=int)

    center = np.array([-0.5, 0.0], dtype=float)
    radius = np.array([aspectRatio, 1], dtype=float)
    start = center - radius
    stop = center + radius

    maxIter = 40
    lastMaxIter = 0
    mod = 40
    cMap = 19

    redoSpace = True
    redoIter = True
    redoImage = True

    key = ord("w")

    cv2.namedWindow("Mandelbrot")
    cv2.setMouseCallback("Mandelbrot", onMouse)

    while True:
        if redoSpace == True:
            c = getComplexPlane(start, stop, resolution)

        if redoIter == True:
            escape = applyNTimes(
                np.zeros_like(c),
                c,
                lastMaxIter,
                maxIter,
                np.zeros_like(c, dtype=float),
            )
        if redoImage == True:
            image = colourImage(escape, mod, cMap)
            cv2.imshow("Mandelbrot", image)
        print(
            f"section [[{start[0]},{start[1]}],[{stop[0]},{stop[1]}]]\nmaxiter {maxIter}"
        )
        print(f"cmap {cMap}\nres {resolution}\n")

        # Föberedelser för nästa iteration
        redoSpace = True
        redoIter = True
        redoImage = True
        lastMaxIter = 0

        key = cv2.waitKey() & 0xFF

        if key == ord("q"):
            cv2.destroyAllWindows()
            break
        elif key == ord("s"):
            x_str = f"{str(start[0]).replace('.','_')}-{str(stop[0]).replace('.','_')}"
            y_str = f"{str(start[1]).replace('.','_')}-{str(stop[1]).replace('.','_')}"
            cv2.imwrite(
                f"output/x-{x_str}-y-{y_str}-m-{maxIter}-c-{cMap}-{mod}.png", image
            )
            redo_space = False
            redo_iter = False
            redo_image = False

        elif key == ord("e"):
            # extend maxiter
            lastMaxIter = maxIter
            maxIter *= 2
            redo_space = False

        elif key == ord("d"):
            # decrease maxiter
            lastMaxIter = 0
            maxIter //= 2
            redo_space = False
            redo_iter = False
            redo_image = False

        elif key == ord("c"):
            # change colormap
            cMap += 1
            cMap %= 22
            redo_space = False
            redo_iter = False

        elif key == ord("z"):
            # zoom in
            radius /= 2
            print("radius", radius)
            start = center - radius
            stop = center + radius

        elif key == ord("x"):
            # zoom out
            radius *= 2
            print("radius", radius)
            start = center - radius
            stop = center + radius

        elif key == ord("+"):
            # increase image size
            k = int(k * 1.2)
            resolution = np.array([aspectRatio * k, k])

        elif key == ord("-"):
            # decrease image size
            k = int(k / 1.2)
            resolution = np.array([aspectRatio * k, k])

        elif key == ord("w"):
            # re-center
            c_point = c[coords[-1][1], coords[-1][0]]
            center[0] = c_point.real
            center[1] = c_point.imag
            start = center - radius
            stop = center + radius
            print("center", center[0], center[1])
