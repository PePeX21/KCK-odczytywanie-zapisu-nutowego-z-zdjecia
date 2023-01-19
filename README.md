# KCK-odczytywanie-zapisu-nutowego-z-zdjecia
## Proces odczytywania informacji z zdjecia

zdjecie orginalne:<br />
<img src="https://user-images.githubusercontent.com/67105405/213574846-ce7a2b46-f85a-4b94-940c-582260efca72.jpg" width="400" height="600"/>
<br />
Przygotowaniu scanu: <br />
-normalizacji <br />
-nałozenia filtru gamma <br />
-odjęciu kolorow <br />
-pozbyciu się cieni z zdjęcia, odjęciu z zdjecia rozswietlonego rozmazanego zdjecia <br />
-przepuszczenie zdjęcia przez threshold
<img src="https://user-images.githubusercontent.com/67105405/213577686-a22829fe-57a0-45ea-b0d8-8873f5fac4c2.jpgwidth" ="400" height="600"/>
<br />
Poszukiwanie krawędzi kartki:<br />
-normalizacji<br />
-nałozenia filtru gamma<br />
-konwersja zdjęcia z BGR do HSV<br />
-eksracja saturacji<br />
-przepuszczenie zdjęcia przez threshold<br />
-znalezienie kontorow kartki i nalozenie obrysowania<br />
<img src="https://user-images.githubusercontent.com/67105405/213579189-56d0132f-9ecd-4a8b-822a-2daf064df6b7.jpg" ="400" height="600"/>
<img src="https://user-images.githubusercontent.com/67105405/213579215-e215d73e-a83a-4db0-ba2e-d343efbc3832.jpg" ="400" height="600"/>
