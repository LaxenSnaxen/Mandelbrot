import cv2
import numpy as np

k = 432
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

key = ord("w")

cv2.namedWindow("Mandelbrot")

# Mus koordinater
coords = []


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        coords.append([x, y])
        print(x, y)


cv2.setMouseCallback("Mandelbrot", onMouse)


def getComplexPlane(start, stop, num):
    # Skaffar arrays från xy arrays
    xSpace = np.linspace(start[0], stop[0], int(num[0]))
    ySpace = np.linspace(start[1], stop[1], int(num[1]))
    # Meshgrids (vet inte vad det betyder)
    real, imag = np.meshgrid(xSpace, ySpace[::-1])
    # Setting upp the komplex plane
    c = np.zeros_like(real, dtype = np.cdouble)
    c.real = real
    c.imag = imag

    return c


def f(z, c):
    return z**2 + c


def applyNTimes(z, c, lastN, n, escape, goOn):
    if not goOn:
        z = np.zeros_like(z)
        escape = np.zeros_like(escape)

    # Applicerar mängd gånger
    for _ in range(lastN, n):
        # Utför grejsimojsen
        z = f(z, c)
        # Kollar vilka positioner som åker iväg
        m = np.abs(z) > 2
        # Increase count for points still in set
        escape[m] += 1

    return z, escape


def colourImage(image, mod, cMap=cv2.COLORMAP_TWILIGHT_SHIFTED):
    # Mandebrot
    mask = image == np.max(image)
    # Modulo bild
    image %= mod
    # Scalear bilden för shcnasiga grejer
    image = (255 / (mod - 1) * image).astype(np.uint8)
    # Färga
    image = cv2.applyColorMap(image, cMap)
    # Färgar mandelbrot svart
    image[mask] = [0, 0, 0]
    return image


def textOverlay(image, center, radius, maxIter, resolution):
    h = resolution[1]
    fontScale = h / 1300
    fontHeight = 30 * fontScale

    c = np.around(center, 3)
    if radius[1] < 1:
        r = [np.format_float_scientific(x, 3, trim="-") for x in radius]
    else:
        r = np.around(radius, 3)
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
    cPoint = c[coords[-1][1], coords[-1][0]]

    center[0] = cPoint.real
    center[1] = cPoint.imag
    start = center - radius
    stop = center + radius

    return center, start, stop


def zoomIn(radius):
    radius /= 2
    start = center - radius
    stop = center + radius

    return radius, start, stop


def zoomOut(radius):
    radius *= 2
    start = center - radius
    stop = center + radius

    return radius, start, stop


def maxIterUp(maxIter):
    lastMaxIter = maxIter
    maxIter *= 2
    return lastMaxIter, maxIter


def maxIterDown(maxIter):
    lastMaxIter = 0
    maxIter //= 2
    return lastMaxIter, maxIter


def saveImage(image, start, stop):
    xStr = f"{str(start[0]).replace('.','_')}-{str(stop[0]).replace('.','_')}"
    yStr = f"{str(start[1]).replace('.','_')}-{str(stop[1]).replace('.','_')}"

    # Sparar bilden i samma mapp som programmet
    cv2.imwrite(f"x-{xStr}-y-{yStr}-m-{maxIter}-c-{cMap}-{mod}.png", image)


def rollcMap(cMap):
    cMap += 1
    cMap %= 22
    return cMap


def resolutionUp(k, aspecRatio):
    k = int(k * 1.2)
    resolution = np.array([aspectRatio * k, k], dtype=int)
    return k, resolution


def resolutionDown(k, aspectRatio):
    k = int(k / 1.2)
    resolution = np.array([aspectRatio * k, k], dtype=int)
    return k, resolution


while True:
    if chr(key) in "wzx+-":
        c = getComplexPlane(start, stop, resolution)
        z = np.zeros_like(c)
        escape = np.zeros_like(c, dtype=float)
    if chr(key) in "wzx+-ed":
        z, escape = applyNTimes(z, c, lastMaxIter, maxIter, escape, chr(key) == "e")
    if chr(key) in "wzx+-efc":
        image = colourImage(escape.copy(), mod, cMap)
        image = textOverlay(image, center, radius, maxIter, resolution)
        cv2.imshow("Mandelbrot", image)

    key = cv2.waitKey() & 0xFF

    if key == ord("q"):  # Quit
        cv2.destroyAllWindows()
        break
    elif key == ord("w"):  # re-center
        center, start, stop = reCenter(coords, c)
    elif key == ord("z"):  # zooma in
        radius, start, stop = zoomIn(radius)
    elif key == ord("x"):  # zooma ut
        radius, start, stop = zoomOut(radius)
    elif key == ord("e"):  # ökar max iter
        lastMaxIter, maxIter = maxIterUp(maxIter)
    elif key == ord("d"):  # Sänker max iter
        lastMaxIter, maxIter = maxIterDown(maxIter)
    elif key == ord("s"):  # spara
        saveImage(image, start, stop)
    elif key == ord("c"):  # ändrar colormap
        cMap = rollcMap(cMap)
    elif key == ord("+"):  # öka bild storlek
        k, resolution = resolutionUp(k, aspectRatio)
    elif key == ord("-"):  # minskar bild storlek
        k, resolution = resolutionDown(k, aspectRatio)
