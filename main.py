import cv2 as cv
import numpy as np
import math
import os

if not os.path.exists('photos'):
    print("there is no photos")
    exit()
photoIterator = 1
print("open photos:")
for photo in os.listdir('photos'):
    print(photo)

    # ---- PATH ----
    pathPhoto = 'photos/' + photo
    pathProcessing = 'processing/' + "photo" + str(photoIterator) + '/'
    if not os.path.exists(pathProcessing):
        os.makedirs(pathProcessing)
    pathResults = 'results/'
    if not os.path.exists(pathResults):
        os.makedirs(pathResults)
    photoIterator += 1

    # ---- PHOTO ----
    trueImg = cv.imread(pathPhoto)
    cv.imwrite(pathProcessing + '1orgin.jpg', trueImg)

    # ---- NORM and GAMMA ----
    norm = np.zeros((800, 800))
    trueImg = cv.normalize(trueImg, norm, 0, 255, cv.NORM_MINMAX)
    cv.imwrite(pathProcessing + '2normalized.jpg', trueImg)

    mid = 0.5
    gamma = math.log(mid * 255) / math.log(np.mean(cv.cvtColor(trueImg, cv.COLOR_BGR2GRAY)))
    trueImg = np.power(trueImg, gamma).clip(0, 255).astype(np.uint8)
    cv.imwrite(pathProcessing + '3gamma.jpg', trueImg)

    # ---- HSV and THRESHOLD ----
    hsvImg = cv.cvtColor(trueImg, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsvImg)
    cv.imwrite(pathProcessing + '4saturation.jpg', s)

    s = cv.normalize(s, norm, 0, 255, cv.NORM_MINMAX)
    cv.imwrite(pathProcessing + '5normalizedSaturation.jpg', s)
    # median = np.median(cv.cvtColor(trueImg, cv.COLOR_BGR2GRAY))
    # _, thresholdImg = cv.threshold(s, (255 - median) / 255 * 100, 255, cv.THRESH_BINARY_INV)  # s 50 255
    _, thresholdImg = cv.threshold(s, 50, 255, cv.THRESH_BINARY_INV)  # s 50 255
    cv.imwrite(pathProcessing + '6threshold.jpg', thresholdImg)

    # ---- SCAN ----
    scanImg = cv.cvtColor(trueImg, cv.COLOR_BGR2GRAY)
    cv.imwrite(pathProcessing + '7gray.jpg', scanImg)
    scanImg = cv.split(scanImg)
    dilatedImg = cv.dilate(scanImg[0], np.ones((7, 7), np.uint8))
    cv.imwrite(pathProcessing + '8dilated.jpg', dilatedImg)
    medianBlurImg = cv.medianBlur(dilatedImg, 21)
    cv.imwrite(pathProcessing + '9medianBlur.jpg', medianBlurImg)
    differenceImg = 255 - cv.absdiff(scanImg[0], medianBlurImg)
    cv.imwrite(pathProcessing + '10noShadows.jpg', differenceImg)
    _, scanImg = cv.threshold(differenceImg, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imwrite(pathProcessing + '11scan.jpg', scanImg)

    # ---- OUTLINE ----
    contours, _ = cv.findContours(thresholdImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    outline = []
    for contour in contours:
        arcLen = cv.arcLength(contour, True)
        if contour[0][0][0] < 2 or contour[0][0][1] < 2 or len(cv.approxPolyDP(contour, 0.1 * arcLen, True)) != 4:
            continue
        else:
            outline.append(contour)
    tmpImg = scanImg.copy()
    cv.drawContours(tmpImg, outline, -1, (0, 0, 255), 20)  # canvas
    cv.imwrite(pathProcessing + '12outLine.jpg', tmpImg)

    # ---- STRAIGHTENING ----
    if len(outline) > 0:
        outline = sorted(outline, key=cv.contourArea)
        cut = outline[-1]

        pts = []
        for i in cv.approxPolyDP(cut, 0.1 * arcLen, True):
            pts.append((i[0][0], i[0][1]))

        pts = np.array(pts, dtype="float32")
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        scanImg = cv.warpPerspective(scanImg, cv.getPerspectiveTransform(rect, dst), (maxWidth, maxHeight))
        cv.imwrite(pathProcessing + '13straightening.jpg', scanImg)
    else:
        print("picture already straight")
    # ---- DETECTING LINES ----

    height, width = scanImg.shape
    horizontalSize = int(width / 15)
    verticalSize = int(height / 400)
    element = np.ones((3, 3))
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontalSize, 1))
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalSize))
    linesImg = cv.dilate(scanImg, horizontalStructure, (-1, -1))
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (int(width / 10), 1))
    linesImg = cv.erode(linesImg, horizontalStructure, (-1, -1))

    scanImg = scanImg[10:-10, :]  # space without 5 lines for papers from ds music side
    linesImg = linesImg[10:-10, :]  # unimportant code
    cv.imwrite(pathProcessing + '14fiveLines.jpg', linesImg)
    # cv.imwrite(pathResults + '19fiveLines' + str(iterator) + '.jpg', binaryImg)

    cannyImg = cv.Canny(linesImg, 10, 100, apertureSize=3)
    cv.imwrite(pathProcessing + '15canny.jpg', cannyImg)

    houghLines = cv.HoughLines(cannyImg, 1, np.pi / 180, 100)
    if houghLines is None:
        print("cant detect lines")
        continue

    all_lines = []
    width, height = cannyImg.shape
    for result_arr in houghLines[:140]:
        rho = result_arr[0][0]
        theta = result_arr[0][1]
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho
        shape_sum = width + height
        x1 = int(x0 + shape_sum * (-b))
        y1 = int(y0 + shape_sum * a)
        x2 = int(x0 - shape_sum * (-b))
        y2 = int(y0 - shape_sum * a)

        start = (x1, y1)
        end = (x2, y2)
        diff = y2 - y1
        if abs(diff) < 10:
            all_lines.append(int((start[1] + end[1]) / 2))

    all_lines = sorted(all_lines)
    detectedLinesImg = linesImg.copy()
    new_lines = []
    for i in all_lines:
        if 50 < i < detectedLinesImg.shape[0] - 50:
            new_lines.append(i)
            cv.line(detectedLinesImg, (0, i), (detectedLinesImg.shape[1] - 1, i), (0, 0, 255))
    cv.imwrite(pathProcessing + '16detectedLines.jpg', detectedLinesImg)

    # ---- CUTTING SCAN INTO LINES ----
    staffs = []  # five lines
    lines = []
    for current_line in all_lines:
        if lines and abs(lines[-1] - current_line) > 50:  # lines distance
            if len(lines) >= 5:
                staffs.append((lines[0], lines[-1]))
            lines.clear()
        lines.append(current_line)

    if len(lines) >= 5:
        if abs(lines[-2] - lines[-1]) <= 50:
            staffs.append((lines[0], lines[-1]))

    resultsImg = []
    iterator = 1
    for staff in staffs:
        linesDistance = int((staff[1] - staff[0]) / 4)  # minimum range = staff[0] maximum  range = staff[1]
        linesLocation = []
        for i in range(5):
            linesLocation.append(staff[0] + i * linesDistance)
        cutImg = scanImg[linesLocation[0] - 55:linesLocation[4] + 55, 5:-5]
        cv.imwrite(pathProcessing + '17cutLines' + str(iterator) + '.jpg', cutImg)

        # ---- SELECTING NOTES ----
        binaryImg = cv.adaptiveThreshold(~cutImg, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
        cv.imwrite(pathProcessing + '18binary.jpg', binaryImg)
        # cv.imwrite(pathResults + '18binary' + str(iterator) + '.jpg', binaryImg)

        height, width = binaryImg.shape
        horizontalSize = int(width / 30)
        verticalSize = int(height / 45)
        horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontalSize, 1))
        verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalSize))
        element = np.ones((2, 2))

        linesImg = cv.erode(binaryImg, horizontalStructure, (-1, -1))
        linesImg = cv.dilate(linesImg, horizontalStructure, (-1, -1))
        cv.imwrite(pathProcessing + '19fiveLines.jpg', linesImg)
        # cv.imwrite(pathResults + '19fiveLines' + str(iterator) + '.jpg', binaryImg)

        differenceImg = binaryImg - linesImg
        differenceImg[differenceImg < 255] = 0
        # cv.imwrite(pathResults + '20binaryAndFiveLinesDifference.jpg', differenceImg)
        cv.imwrite(pathProcessing + '20binaryAndFiveLinesDifference' + str(iterator) + '.jpg', differenceImg)

        differenceImg = cv.erode(differenceImg, element)
        differenceImg = cv.dilate(differenceImg, element)
        # cv.imwrite(pathResults + '21filteredResults1.jpg', binaryImg)
        cv.imwrite(pathProcessing + '21filteredResults1' + str(iterator) + '.jpg', differenceImg)

        fiveLinesGapMakerImg = cv.dilate(differenceImg, verticalStructure, (-1, -1))
        fiveLinesGapMakerImg = cv.dilate(fiveLinesGapMakerImg, verticalStructure, (-1, -1))
        fiveLinesGapMakerImg = cv.dilate(fiveLinesGapMakerImg, verticalStructure, (-1, -1))
        # cv.imwrite(pathResults + '22gapMaker.jpg', fiveLinesGapMakerImg)
        cv.imwrite(pathProcessing + '22gapMaker' + str(iterator) + '.jpg', fiveLinesGapMakerImg)

        linesImg = linesImg - fiveLinesGapMakerImg
        linesImg[linesImg < 255] = 0
        cv.imwrite(pathProcessing + '23FiveLinesWithGaps.jpg', linesImg)

        binaryImg = binaryImg - linesImg
        binaryImg[binaryImg < 255] = 0
        # cv.imwrite(pathResults + '24binaryAndFiveLinesDifference2.jpg', binaryImg)
        cv.imwrite(pathProcessing + '24binaryAndFiveLinesDifference2' + str(iterator) + '.jpg', binaryImg)

        binaryImg = cv.erode(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        # cv.imwrite(pathResults + '25filteredResults2.jpg', binaryImg)
        cv.imwrite(pathProcessing + '25filteredResults2' + str(iterator) + '.jpg', binaryImg)

        horizontalSize = int(width / 300)
        horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontalSize, 1))
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.erode(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.dilate(binaryImg, horizontalStructure, (-1, -1))
        binaryImg = cv.erode(binaryImg, element)
        # cv.imwrite(pathResults + '26processBinary.jpg', binaryImg)
        cv.imwrite(pathProcessing + '26processBinary' + str(iterator) + '.jpg', binaryImg)

        separatedImg = cv.bitwise_not(binaryImg)
        cv.imwrite(pathProcessing + '27separatedNotes' + str(iterator) + '.jpg', separatedImg)

        # ---- NOTE CONTOURS ----
        scale_percent = 300  # because findContours is to stupid ...
        width = int(separatedImg.shape[1] * scale_percent / 100)
        height = int(separatedImg.shape[0] * scale_percent / 100)
        dim = (width, height)
        separatedImg = cv.resize(separatedImg, dim, interpolation=cv.INTER_AREA)  # resize image

        arrayOfLinesHeight = [i for i in range(55, 55 + linesDistance * 4 + 1, linesDistance)]
        arrayOfLinesHeight = np.multiply(arrayOfLinesHeight, 3)  # after canny mistake
        linesDistance = 3 / 4 * linesDistance
        arrayOfLinesHeight = arrayOfLinesHeight          # 15 was in another alternative

        contours, hierarchy = cv.findContours(separatedImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda k: k[0][0][0])
        arrayOfNotes = []
        keyBot = height / 2
        stickyFrame = True

        for cnt in range(1, len(contours)):
            contour = contours[cnt]

            topMost = tuple(contour[contour[:, :, 1].argmin()][0])[1]
            botMost = tuple(contour[contour[:, :, 1].argmax()][0])[1]
            leftMost = tuple(contour[contour[:, :, 0].argmin()][0])[0]
            rightMost = tuple(contour[contour[:, :, 0].argmax()][0])[0]

            if contour[0][0][0] <= int(width / 8 - width / 20):
                if botMost - topMost > 100:
                    stickyFrame = False  # if true then violin get stick
                    keyBot = botMost
                continue

            if topMost == 0:  # frame from hell
                continue

            if botMost - topMost > 45 and contour[0][0][0] > int(width / 8 - width / 20):
                x = [contour[i][0][0] for i in range(len(contour))]
                y = [contour[i][0][1] for i in range(len(contour))]
                nl = max([min(x), 0])
                nr = min([max(x), separatedImg.shape[1] - 1])
                nd = max([min(y), 0])
                nu = min([max(y), separatedImg.shape[0] - 1])

                cv.rectangle(separatedImg, (nl, nu), (nr, nd), (0, 0, 255), 3)
                cv.imwrite(pathProcessing + '28rectangle' + str(iterator) + '.jpg', separatedImg)

                # ---- DETECTING POSITION ----

                # for coordinate in contour:        # both direction
                #    if coordinate[0][0] == rightMost:
                #        if coordinate[0][1] < (topMost+botMost)/2:
                #            positionOfNote = [(rightMost + leftMost)/2, botMost - 30]
                #            break
                #        else:
                #            positionOfNote = [(rightMost + leftMost)/2, topMost + 30]
                #            break
                # positionOfNote = [(rightMost + leftMost)/2, botMost - 30]

                # avg = 0
                # for coordinate in contour:        # both direction
                #    if coordinate[0][0] == leftMost:
                #       avg = avg + coordinate[0][1]
                #
                # positionOfNote = [(rightMost + leftMost)/2, avg/2 - 11]

                positionOfNote = [(rightMost + leftMost) / 2, botMost - (arrayOfLinesHeight[1] - arrayOfLinesHeight[0]) / 1.5]
                positionOfNote[0] = int(positionOfNote[0])  # because of tuple
                positionOfNote[1] = int(positionOfNote[1])

                pitch = ["G", "H"]
                if arrayOfLinesHeight[0] - linesDistance < positionOfNote[1]:
                    pitch = ["F", "A"]
                if arrayOfLinesHeight[0] + linesDistance < positionOfNote[1]:
                    pitch = ["E", "G"]
                if arrayOfLinesHeight[1] - linesDistance < positionOfNote[1]:
                    pitch = ["D", "F"]
                if arrayOfLinesHeight[1] + linesDistance < positionOfNote[1]:
                    pitch = ["C", "E"]
                if arrayOfLinesHeight[2] - linesDistance < positionOfNote[1]:
                    pitch = ["H", "D"]
                if arrayOfLinesHeight[2] + linesDistance < positionOfNote[1]:
                    pitch = ["A", "C"]
                if arrayOfLinesHeight[3] - linesDistance < positionOfNote[1]:
                    pitch = ["G", "H"]
                if arrayOfLinesHeight[3] + linesDistance < positionOfNote[1]:
                    pitch = ["F", "A"]
                if arrayOfLinesHeight[4] - linesDistance < positionOfNote[1]:
                    pitch = ["E", "G"]
                if arrayOfLinesHeight[4] + linesDistance < positionOfNote[1]:
                    pitch = ["D", "F"]
                positionOfNote.append(pitch)
                arrayOfNotes.append(positionOfNote)

                # print("\npozycja ", positionOfNote)
                # print("KEY ", pitch)
                # print("odleglosc liniowa ", linesDistance)
                # print("te linie ", arrayOfLinesHeight)

        if len(arrayOfNotes) == 0:
            continue

        whole = width - int(width / 8 - width / 32) - arrayOfNotes[0][0]
        half = whole / 2
        quarter = half / 2
        eight = quarter / 2
        for i in range(len(arrayOfNotes)):
            if i + 1 == len(arrayOfNotes):
                value = "whole"
                distance = width - int(width / 8 - width / 32) - arrayOfNotes[i][0]
                if distance < whole - quarter:
                    value = "half"
                if distance < half - eight:
                    value = "quarter"
                if distance < quarter - eight / 2:
                    value = "eight"
            else:
                distance = arrayOfNotes[i + 1][0] - arrayOfNotes[i][0]
                value = "half"
                if distance < half - eight:
                    value = "quarter"
                if distance < quarter - eight / 2:
                    value = "eight"
            arrayOfNotes[i].append(value)

        # ---- SHOWING RESULTS OF DETECTING ----
        tmpImg = cv.resize(cutImg, dim, interpolation=cv.INTER_AREA)  # to corresponded with
        for i in arrayOfNotes:
            tmpImg = cv.circle(tmpImg, (i[0] + 40, i[1]), radius=15, color=(0, 0, 255), thickness=-1)
        tmpImg = cv.circle(tmpImg, (int(width / 8 - width / 20), arrayOfLinesHeight[0]), radius=15, color=(0, 0, 255), thickness=-1)
        tmpImg = cv.circle(tmpImg, (int(width / 8 - width / 20), arrayOfLinesHeight[1]), radius=15, color=(0, 0, 255), thickness=-1)
        tmpImg = cv.circle(tmpImg, (int(width / 8 - width / 20), arrayOfLinesHeight[2]), radius=15, color=(0, 0, 255), thickness=-1)
        tmpImg = cv.circle(tmpImg, (int(width / 8 - width / 20), arrayOfLinesHeight[3]), radius=15, color=(0, 0, 255), thickness=-1)
        tmpImg = cv.circle(tmpImg, (int(width / 8 - width / 20), arrayOfLinesHeight[4]), radius=15, color=(0, 0, 255), thickness=-1)

        cv.imwrite(pathProcessing + '29linesAndNotesPosition' + str(iterator) + '.jpg', tmpImg)

        tmpImg = cv.resize(cutImg, dim, interpolation=cv.INTER_AREA)  # to corresponded with
        font = cv.FONT_HERSHEY_SIMPLEX
        if stickyFrame:
            cv.putText(tmpImg, "treble", (125, arrayOfLinesHeight[4] + 120), font, 2, (0, 0, 255), 2, cv.LINE_AA)
            for i in arrayOfNotes:
                cv.putText(tmpImg, i[2][0], (i[0], arrayOfLinesHeight[4] + 75), font, 2, (0, 0, 255), 2, cv.LINE_AA)
                cv.putText(tmpImg, i[3], (i[0], arrayOfLinesHeight[4] + 135), font, 2, (0, 0, 255), 2, cv.LINE_AA)
        else:
            if keyBot > arrayOfLinesHeight[4] + 11:
                cv.putText(tmpImg, "treble", (125, arrayOfLinesHeight[4] + 120), font, 2, (0, 0, 255), 2, cv.LINE_AA)
                for i in arrayOfNotes:
                    cv.putText(tmpImg, i[2][0], (i[0], arrayOfLinesHeight[4] + 75), font, 2, (0, 0, 255), 2, cv.LINE_AA)
                    cv.putText(tmpImg, i[3], (i[0], arrayOfLinesHeight[4] + 135), font, 2, (0, 0, 255), 2, cv.LINE_AA)
            else:
                cv.putText(tmpImg, "bass", (125, arrayOfLinesHeight[4] + 120), font, 2, (0, 0, 255), 2, cv.LINE_AA)
                for i in arrayOfNotes:
                    cv.putText(tmpImg, i[2][1], (i[0], arrayOfLinesHeight[4] + 75), font, 2, (0, 0, 255), 2, cv.LINE_AA)
                    cv.putText(tmpImg, i[3], (i[0], arrayOfLinesHeight[4] + 135), font, 2, (0, 0, 255), 2, cv.LINE_AA)
        cv.imwrite(pathProcessing + '30writtenInfo' + str(iterator) + '.jpg', tmpImg)
        resultsImg.append(tmpImg)
        iterator += 1

    # ---- WRITING RESULTS ----
    endImg = np.concatenate((resultsImg[0], resultsImg[1]), axis=0)
    for i in range(2, len(resultsImg)):
        endImg = np.concatenate((endImg, resultsImg[i]), axis=0)
    cv.imwrite(pathProcessing + '0Results.jpg', endImg)
    cv.imwrite(pathResults + 'Results' + str(photoIterator - 1) + '.jpg', endImg)

# ogarnac laski w druga strone detekcja czy istnieje ktos skrany


'''
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@ WERSJE DETEKCJI @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# 1
# bez zadnego ogolnego erote + vertical na 45
        binaryImg = cv.adaptiveThreshold(~cutImg, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
        cv.imwrite(pathResults + '18binary.jpg', binaryImg)
        # cv.imwrite(pathResults + '18binary' + str(iterator) + '.jpg', binaryImg)

        height, width = binaryImg.shape
        horizontalSize = int(width / 45)        # ZMIENIONE !!!!!!!!!!!!!!!!!!!!!!!!! bylo 30
        verticalSize = int(height / 45)
        horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontalSize, 1))
        verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalSize))

        element = np.ones((2, 2))

        # bez zadnego ogolnego erote + vertical na 45
        binaryImg = cv.erode(binaryImg, verticalStructure, (-1, -1))
        # cv.imwrite(pathResults + '20EroVEBinary1' + str(iterator) + '.jpg', binaryImg)
        cv.imwrite(pathResults + '20EroVEBinary1.jpg', binaryImg)
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.erode(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.erode(binaryImg, verticalStructure, (-1, -1))
        # cv.imwrite(pathResults + '20EroVEBinary2' + str(iterator) + '.jpg', binaryImg)
        cv.imwrite(pathResults + '20EroVEBinary2.jpg', binaryImg)
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.erode(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.erode(binaryImg, verticalStructure, (-1, -1))
        # cv.imwrite(pathResults + '20EroVEBinary3' + str(iterator) + '.jpg', binaryImg)
        cv.imwrite(pathResults + '20EroVEBinary3.jpg', binaryImg)
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.erode(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.erode(binaryImg, verticalStructure, (-1, -1))
        # cv.imwrite(pathResults + '21DilateBinary' + str(iterator) + '.jpg', binaryImg)
        cv.imwrite(pathResults + '20EroVEBinary4.jpg', binaryImg)
        binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.dilate(binaryImg, horizontalStructure, (-1, -1))
        # cv.imwrite(pathResults + '21DilateBinary' + str(iterator) + '.jpg', binaryImg)
        cv.imwrite(pathResults + '21DilateBinary.jpg', binaryImg)

        separatedImg = cv.bitwise_not(binaryImg)
        cv.imwrite(pathResults + '23separatedNotes' + str(iterator) + '.jpg', separatedImg)
        iterator += 1

'''

'''
# 2
        binaryImg = cv.adaptiveThreshold(~cutImg, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
        cv.imwrite(pathResults + '18binary.jpg', binaryImg)
        # cv.imwrite(pathResults + '18binary' + str(iterator) + '.jpg', binaryImg)

        height, width = binaryImg.shape
        horizontalSize = int(width / 30)
        verticalSize = int(height / 45)
        horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontalSize, 1))
        verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalSize))

        eroH = cv.erode(binaryImg, horizontalStructure, (-1, -1))
        dilH = cv.dilate(eroH, horizontalStructure, (-1, -1))
        cv.imwrite(pathResults + '19fiveLines.jpg', dilH)
        # cv.imwrite(pathResults + '19fiveLines' + str(iterator) + '.jpg', binaryImg)

        binaryImg = binaryImg - dilH
        cv.imwrite(pathResults + '20binaryAndFiveLinesDifference.jpg', binaryImg)
        # cv.imwrite(pathResults + '20binaryAndFiveLinesDifference' + str(iterator) + '.jpg', binaryImg)

        # ktotkie
        # vertical 30
        # binaryImg = cv.erode(binaryImg, verticalStructure, (-1, -1))
        # cv.imwrite(pathResults + '21processBinary1.jpg', binaryImg)
        # binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        # binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        #element = np.ones((3, 3))
        # binaryImg = cv.dilate(binaryImg, element)
        # cv.imwrite(pathResults + '22processBinary1.jpg', binaryImg)

        element = np.ones((2, 2))
        binaryImg = cv.erode(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        cv.imwrite(pathResults + '21filteredResults.jpg', binaryImg)
        # cv.imwrite(pathResults + '19improvedBinary' + str(iterator) + '.jpg', binaryImg)

        binaryImg = cv.erode(binaryImg, verticalStructure, (-1, -1))
        cv.imwrite(pathResults + '22processBinary1.jpg', binaryImg)
        # cv.imwrite(pathResults + '20EroVEBinary1' + str(iterator) + '.jpg', binaryImg)

        # jak bd juz zdjecia i bd zle te 2 linijki ponizej wywalic
        # a jak bardzo zle erota z gory tez

        binaryImg = cv.erode(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        cv.imwrite(pathResults + '22processBinary2.jpg', binaryImg)
        # cv.imwrite(pathResults + '20EroVEBinary2' + str(iterator) + '.jpg', binaryImg)

        binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        cv.imwrite(pathResults + '22processBinary3.jpg', binaryImg)
        # cv.imwrite(pathResults + '22processEBinary3' + str(iterator) + '.jpg', binaryImg)

        separatedImg = cv.bitwise_not(binaryImg)
        cv.imwrite(pathResults + '23separatedNotes' + str(iterator) + '.jpg', separatedImg)
        iterator += 1
'''

'''
# 4
        ################## short alternative #####################
        
        eroH = cv.erode(binaryImg, horizontalStructure, (-1, -1))
        dilH = cv.dilate(eroH, horizontalStructure, (-1, -1))
        cv.imwrite(pathResults + '19fiveLines.jpg', dilH)
        # cv.imwrite(pathResults + '19fiveLines' + str(iterator) + '.jpg', binaryImg)

        binaryImg = binaryImg - dilH
        cv.imwrite(pathResults + '20binaryAndFiveLinesDifference.jpg', binaryImg)
        # cv.imwrite(pathResults + '20binaryAndFiveLinesDifference' + str(iterator) + '.jpg', binaryImg)
        
        # vertical 30
        binaryImg = cv.erode(binaryImg, verticalStructure, (-1, -1))
        cv.imwrite(pathResults + '21processBinary1.jpg', binaryImg)
        binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        element = np.ones((3, 3))
        binaryImg = cv.dilate(binaryImg, element)
        cv.imwrite(pathResults + '22processBinary1.jpg', binaryImg)
'''

'''
# 4 agrrrrrrrrrrrrr
        binaryImg = cv.adaptiveThreshold(~cutImg, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
        cv.imwrite(pathResults + '18binary.jpg', binaryImg)
        # cv.imwrite(pathResults + '18binary' + str(iterator) + '.jpg', binaryImg)

        height, width = binaryImg.shape
        horizontalSize = int(width / 30)
        verticalSize = int(height / 45)
        horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontalSize, 1))
        verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalSize))
        element = np.ones((2, 2))

        eroH = cv.erode(binaryImg, horizontalStructure, (-1, -1))
        dilH = cv.dilate(eroH, horizontalStructure, (-1, -1))
        cv.imwrite(pathResults + '19fiveLines.jpg', dilH)
        # cv.imwrite(pathResults + '19fiveLines' + str(iterator) + '.jpg', binaryImg)

        differenceImg = np.setdiff1d(binaryImg, dilH, assume_unique=False)
        cv.imwrite(pathResults + '20binaryAndFiveLinesDifference.jpg', differenceImg)
        # cv.imwrite(pathResults + '20binaryAndFiveLinesDifference' + str(iterator) + '.jpg', binaryImg)

        fiveLinesGapMakerImg = cv.erode(differenceImg, element)
        fiveLinesGapMakerImg = cv.dilate(fiveLinesGapMakerImg, element)

        fiveLinesGapMakerImg = cv.dilate(fiveLinesGapMakerImg, verticalStructure, (-1, -1))
        fiveLinesGapMakerImg = cv.dilate(fiveLinesGapMakerImg, verticalStructure, (-1, -1))
        fiveLinesGapMakerImg = cv.dilate(fiveLinesGapMakerImg, verticalStructure, (-1, -1))
        fiveLinesGapMakerImg = cv.dilate(fiveLinesGapMakerImg, verticalStructure, (-1, -1))
        fiveLinesGapMakerImg = cv.dilate(fiveLinesGapMakerImg, element)
        fiveLinesGapMakerImg = cv.dilate(fiveLinesGapMakerImg, element)
        cv.imwrite(pathResults + '21gapMaker.jpg', fiveLinesGapMakerImg)
        # cv.imwrite(pathResults + '21filteredResults' + str(iterator) + '.jpg', binaryImg)

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11dilh przed:")
        for i in dilH:
            for j in i:
                print(type(dilH), "echh i i: ", type(i), "iii j:", type(j))
                if j == 1:
                    print(j)
                    print("nie rozumiemnp.")
        dilH = dilH - fiveLinesGapMakerImg
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!dilh po:")
        for i in dilH:
            for j in i:
                print(type(dilH), "echh i i: " , type(i), "iii j:", type(j))
                if j == 1:
                    print("kurwa co jest")
                    j = 255;
        cv.imwrite(pathResults + '22FiveLinesWithGaps.jpg', dilH)
        cv.imwrite(pathResults + '22AAAAAchuj CI w dupe.jpg', binaryImg)
        binaryImg = binaryImg - dilH

        cv.imwrite(pathResults + '23binaryAndFiveLinesDifference2.jpg', binaryImg)
        # cv.imwrite(pathResults + '20binaryAndFiveLinesDifference' + str(iterator) + '.jpg', binaryImg)

        binaryImg = cv.erode(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        cv.imwrite(pathResults + '24filteredResults.jpg', binaryImg)
        # cv.imwrite(pathResults + '19improvedBinary' + str(iterator) + '.jpg', binaryImg)

        binaryImg = cv.erode(binaryImg, verticalStructure, (-1, -1))
        cv.imwrite(pathResults + '25processBinary1.jpg', binaryImg)
        # cv.imwrite(pathResults + '20EroVEBinary1' + str(iterator) + '.jpg', binaryImg)

        # jak bd juz zdjecia i bd zle te 2 linijki ponizej wywalic
        # a jak bardzo zle erota z gory tez

        binaryImg = cv.erode(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        cv.imwrite(pathResults + '25processBinary2.jpg', binaryImg)
        # cv.imwrite(pathResults + '20EroVEBinary2' + str(iterator) + '.jpg', binaryImg)

        binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        cv.imwrite(pathResults + '25processBinary3.jpg', binaryImg)
        # cv.imwrite(pathResults + '22processEBinary3' + str(iterator) + '.jpg', binaryImg)

        separatedImg = cv.bitwise_not(binaryImg)
        cv.imwrite(pathResults + '26separatedNotes' + str(iterator) + '.jpg', separatedImg)
        iterator += 1
'''

'''
# polepszone 4
        binaryImg = cv.adaptiveThreshold(~cutImg, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
        cv.imwrite(pathResults + '18binary.jpg', binaryImg)
        # cv.imwrite(pathResults + '18binary' + str(iterator) + '.jpg', binaryImg)

        height, width = binaryImg.shape
        horizontalSize = int(width / 30)
        verticalSize = int(height / 45)
        horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontalSize, 1))
        verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalSize))

        eroH = cv.erode(binaryImg, horizontalStructure, (-1, -1))
        dilH = cv.dilate(eroH, horizontalStructure, (-1, -1))
        cv.imwrite(pathResults + '19fiveLines.jpg', dilH)
        # cv.imwrite(pathResults + '19fiveLines' + str(iterator) + '.jpg', binaryImg)

        binaryImg = binaryImg - dilH
        cv.imwrite(pathResults + '20binaryAndFiveLinesDifference.jpg', binaryImg)
        # cv.imwrite(pathResults + '20binaryAndFiveLinesDifference' + str(iterator) + '.jpg', binaryImg)

        element = np.ones((2, 2))
        binaryImg = cv.erode(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        cv.imwrite(pathResults + '21filteredResults.jpg', binaryImg)
        # cv.imwrite(pathResults + '19improvedBinary' + str(iterator) + '.jpg', binaryImg)

        binaryImg = cv.erode(binaryImg, verticalStructure, (-1, -1))
        cv.imwrite(pathResults + '22processBinary1.jpg', binaryImg)
        # cv.imwrite(pathResults + '20EroVEBinary1' + str(iterator) + '.jpg', binaryImg)

        #binaryImg = cv.erode(binaryImg, element)
        #binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.erode(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.erode(binaryImg, verticalStructure, (-1, -1))
        cv.imwrite(pathResults + '22processBinary2.jpg', binaryImg)
        # cv.imwrite(pathResults + '20EroVEBinary2' + str(iterator) + '.jpg', binaryImg)

        #binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        #binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        #binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        #binaryImg = cv.dilate(binaryImg, verticalStructure, (-1, -1))
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        binaryImg = cv.dilate(binaryImg, element)
        cv.imwrite(pathResults + '22processBinary3.jpg', binaryImg)
        # cv.imwrite(pathResults + '22processEBinary3' + str(iterator) + '.jpg', binaryImg)

        separatedImg = cv.bitwise_not(binaryImg)
        cv.imwrite(pathResults + '23separatedNotes' + str(iterator) + '.jpg', separatedImg)
        iterator += 1

'''
