# KCK-odczytywanie-zapisu-nutowego-z-zdjecia
## Proces odczytywania informacji z zdjecia

zdjecie orginalne:
![1orgin](https://user-images.githubusercontent.com/67105405/213574846-ce7a2b46-f85a-4b94-940c-582260efca72.jpg)

Przygotowaniu scanu:\\
-normalizacji \\
-nałozenia filtru gamma \\
-odjęciu kolorow
-pozbyciu się cieni z zdjęcia, odjęciu z zdjecia rozswietlonego rozmazanego zdjecia
-przepuszczenie zdjęcia przez threshold
![11scan](https://user-images.githubusercontent.com/67105405/213577686-a22829fe-57a0-45ea-b0d8-8873f5fac4c2.jpg)

Poszukiwanie krawędzi kartki:
-normalizacji
-nałozenia filtru gamma
-konwersja zdjęcia z BGR do HSV
-eksracja saturacji
-przepuszczenie zdjęcia przez threshold
-znalezienie kontorow kartki i nalozenie obrysowania
![6threshold](https://user-images.githubusercontent.com/67105405/213579189-56d0132f-9ecd-4a8b-822a-2daf064df6b7.jpg)
![12outLine](https://user-images.githubusercontent.com/67105405/213579215-e215d73e-a83a-4db0-ba2e-d343efbc3832.jpg)
