import cv2
import numpy as np

coords = []  # Här sparas koordinaterna som användaren klickar på


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Om vänster musknapp klickas
        coords.append([x, y])
        print(x, y)


def getComplexPlane(start, stop, num):
    # Skaffar arrays från xy arrays
    xSpace = np.linspace(start[0], stop[0], int(num[0]))
    ySpace = np.linspace(start[1], stop[1], int(num[1]))
    # Meshgrids (det mellan det riktiga och komplexa nummer, en mesh av de båda)
    real, imag = np.meshgrid(xSpace, ySpace[::-1]) # Meshgrid skapar en grid av de två arrayerna, en array med eindimensionala koordinater.
    # Bygger den komplexa planan
    c = np.zeros_like(real, dtype=np.cdouble)
    c.real = real
    c.imag = imag

    return c


def f(z, c):
    return z**2 + c  # beräkningen för mandelbrot


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


# Lägger till text på bilden
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
    # Lägger till texten
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

if __name__ == "__main__":

    k = 600 # Påverkar upplösningen, storleken av fönstret. i aspect ratio 16/9 så är 600: 9.
    aspectRatio = 16 / 9
    resolution = np.array([aspectRatio * k, k], dtype=int)  # Här räknar programmet ut upplösningen.

    center = np.array([-0.5, 0.0], dtype=float) # Här är centrumet av bilden
    radius = np.array([aspectRatio, 1], dtype=float) # Här är radien av bilden
    start = center - radius # Här är starten av bilden
    stop = center + radius # Här är slutet av bilden, programmet gör det här för att kolla vart gränserna går.

    maxIter = 40 # Här är antalet iterationer till at börja med.
    lastMaxIter = 0 # Här är antalet iterationer från förra gången.
    mod = 40 # Här är modulot av bilden.
    cMap = 19 # Här är färgkartan av bilden.

    redoSpace = True # Om redoSpace är True så gör den om hela skiten, eller skapar den.
    redoIter = True # Om redoIter är True så gör den om iterationerna.
    redoImage = True # Om redoImage är True så gör den om bilden.
    # De olika boolean värdena ovanför ändras beroende på vad användaren gör. Om användaren spara bilden ska den inte slösa 


    cv2.namedWindow("Mandelbrot")
    cv2.setMouseCallback("Mandelbrot", onMouse) # Kollar vart musen klickar i programmet.

    while True:
        if redoSpace == True:
            c = getComplexPlane(start, stop, resolution)

        if redoIter == True:
            escape = applyNTimes(
                np.zeros_like(c), # zeros_like skapar en array med samma form som c
                c,
                lastMaxIter,
                maxIter,
                np.zeros_like(c, dtype=float), # Här dikterar dtype=float vilken data typ som man får tillbaka av zeros_like
            )
        if redoImage == True:
            image = textOverlay(colourImage(escape, mod, cMap), center, radius, maxIter, resolution) # Här lägger den till texten på bilden och färgar den.
            cv2.imshow("Mandelbrot", image) # Bilden visas
        print(
            f"section [[{start[0]},{start[1]}],[{stop[0]},{stop[1]}]]\nmaxiter {maxIter}" # Information om bilden skrivs ut.
        )
        print(f"cmap {cMap}\nres {resolution}\n") # Ytterligare information skrivs ut.

        # Föberedelser för nästa iteration
        redoSpace = True
        redoIter = True
        redoImage = True
        lastMaxIter = 0

        key = cv2.waitKey() & 0xFF # 0xFF gör så att rätt nummer kommer till wailKey, annars kan det bli fel. För att om man har num lock återvänder olika nummer.

        if key == ord("q"):
            cv2.destroyAllWindows() # Stänger alla fönster och avslutar loopen.
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
            # Ökar Maxiter, ger fler detaljer.
            lastMaxIter = maxIter
            maxIter *= 2
            redo_space = False

        elif key == ord("d"):
            # Minskar Maxiter, ger färre detaljer.
            lastMaxIter = 0
            maxIter //= 2
            redo_space = False
            redo_iter = False
            redo_image = False

        elif key == ord("c"):
            # Ändrar colormap, vilket ändrar färgerna.
            cMap += 1
            cMap %= 22
            redo_space = False
            redo_iter = False

        elif key == ord("z"):
            # Zooma in
            radius /= 2
            print("radius", radius)
            start = center - radius
            stop = center + radius

        elif key == ord("x"):
            # Zooma ut
            radius *= 2
            print("radius", radius)
            start = center - radius
            stop = center + radius

        elif key == ord("+"):
            # Ökar storleken av bilden på användarens skärm.
            k = int(k * 1.2)
            resolution = np.array([aspectRatio * k, k])

        elif key == ord("-"):
            # Minskar storleken av bilden på användarens skärm.
            k = int(k / 1.2)
            resolution = np.array([aspectRatio * k, k])

        elif key == ord("w"):
            # Centrerar bilden på användarens skärm, från där användaren klickar.
            c_point = c[coords[-1][1], coords[-1][0]]
            center[0] = c_point.real
            center[1] = c_point.imag
            start = center - radius
            stop = center + radius
            print("center", center[0], center[1])
