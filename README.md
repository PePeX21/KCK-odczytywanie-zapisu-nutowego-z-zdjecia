# KCK-odczytywanie-zapisu-nutowego-z-zdjecia
## Proces odczytywania informacji z zdjecia

zdjecie orginalne:<br />
<img src="https://user-images.githubusercontent.com/67105405/213574846-ce7a2b46-f85a-4b94-940c-582260efca72.jpg" width="400" height="600"/>
<br />
<br />
Przygotowaniu scanu: <br />
-normalizacji <br />
-nałozenia filtru gamma <br />
-odjęciu kolorow <br />
-pozbyciu się cieni z zdjęcia, odjęciu z zdjecia rozswietlonego rozmazanego zdjecia <br />
-przepuszczenie zdjęcia przez threshold <br />
<img src="https://user-images.githubusercontent.com/67105405/213577686-a22829fe-57a0-45ea-b0d8-8873f5fac4c2.jpg" width ="400" height="600"/>
<br />
<br />
 Poszukiwanie krawędzi kartki:<br />
-normalizacji<br />
-nałozenia filtru gamma<br />
-konwersja zdjęcia z BGR do HSV<br />
-eksracja saturacji<br />
-przepuszczenie zdjęcia przez threshold<br />
-znalezienie kontorow kartki i nalozenie obrysowania<br />
<img src="https://user-images.githubusercontent.com/67105405/213579189-56d0132f-9ecd-4a8b-822a-2daf064df6b7.jpg" width ="400" height="600"/>
<br />
<br />
Nałożenie obrysowania wycięcie obrazu<br />
<img src="https://user-images.githubusercontent.com/67105405/213579215-e215d73e-a83a-4db0-ba2e-d343efbc3832.jpg" width ="400" height="600"/>
<img src="https://user-images.githubusercontent.com/67105405/213583151-c2af7284-fcce-41d1-a19a-41a0a2bcd1cf.jpg" width ="400" height="600"/>
<br />
<br />
Usuniecie znakow zapisu nutowego z uzyciem erozji oraz poszukiwanie lini z uzyciem filtru canny<br />
<img src="https://user-images.githubusercontent.com/67105405/213583643-286b6b98-8ced-491a-8088-ccf7ceb9acd2.jpg" width ="400" height="600"/>
<img src="https://user-images.githubusercontent.com/67105405/213583659-a99aa1d8-1022-497f-81ed-2999d79316b3.jpg" width ="400" height="600"/>
<br />
<br />
Wycinanie pięciolini w oparciu o znalezione linie:
<img src="https://user-images.githubusercontent.com/67105405/213583881-c3b47ab9-3835-446f-bbec-4e6bf4b044cb.jpg" width ="400" height="80"/>
<img src="https://user-images.githubusercontent.com/67105405/213583884-d4b738ee-3cb4-4bbb-968b-796670023f09.jpg" width ="400" height="80"/>
<img src="https://user-images.githubusercontent.com/67105405/213583888-8cfef7cd-998b-437d-8281-efd149404fe8.jpg" width ="400" height="80"/>
<img src="https://user-images.githubusercontent.com/67105405/213583890-b8d2c129-a1e1-426a-9b4f-08a9672b96f3.jpg" width ="400" height="80"/>
<img src="https://user-images.githubusercontent.com/67105405/213583893-d7b7f589-1509-4269-80ac-fc8bdb7d65af.jpg" width ="400" height="80"/>
<img src="https://user-images.githubusercontent.com/67105405/213583896-18bc3f9b-2bb1-420d-a5e8-afa57e1f2f26.jpg" width ="400" height="80"/>
<img src="https://user-images.githubusercontent.com/67105405/213583898-3ba04882-47b3-48c5-b135-8e572fee6be1.jpg" width ="400" height="80"/>
<br />
<br />
Wycinanie pojedynczych nut:<br />
- stworzenie obrazu binarnego<br />
- znalezienie lini z uzyciem erozji i rozrostu<br />
- odjecie lini z obrazka<br />
- powiekszenie nut z uzyciem erozji<br />
- usuniecie z obrazu binarnego powiekszonych nut<br />
- porownanie obrazu binarnego z obrazem zawierajacym puste przestrzenie po usunieciu nut<br />
<img src="https://user-images.githubusercontent.com/67105405/213584626-c8107c42-3ad8-4626-baab-75cb6163a726.jpg" width ="400" height="80"/>
<img src="https://user-images.githubusercontent.com/67105405/213584654-b5f89929-ac90-4188-a331-ed092dad40f1.jpg" width ="400" height="80"/>
<img src="https://user-images.githubusercontent.com/67105405/213584713-3bbc648f-f277-4167-b020-b1f0542b859a.jpg" width ="400" height="80"/>
<img src="https://user-images.githubusercontent.com/67105405/213584934-4751e7f1-9a33-480f-b8f5-974c416f7c4c.jpg" width ="400" height="80"/>
<img src="https://user-images.githubusercontent.com/67105405/213584992-0d4537c5-3eec-42a0-8f2d-743f8256c182.jpg" width ="400" height="80"/>
<img src="https://user-images.githubusercontent.com/67105405/213585029-e7f8af92-0826-4568-ac8a-0541b7b9de7a.jpg" width ="400" height="80"/>
<img src="https://user-images.githubusercontent.com/67105405/213585056-2f2fc2fd-a7bd-48d7-a0d6-89c3faec0341.jpg" width ="400" height="80"/>
<br />
<br />
Detekcja nut<br />
Odniesienie do pozycji na pieciolini<br />
predykcja znaku w oparciu o wysokosc i odleglosc do kolejnego znaku<br />
<img src="https://user-images.githubusercontent.com/67105405/213585102-9498e973-d5a8-4cda-9b74-decac80ee00f.jpg" width ="400" height="80"/>
<img src="https://user-images.githubusercontent.com/67105405/213585164-551b0932-15b0-49b6-a7af-b1440b8df57c.jpg" width ="400" height="80"/>
<img src="https://user-images.githubusercontent.com/67105405/213585174-3199809d-0d66-40ee-bcb2-15f57c64b803.jpg" width ="400" height="80"/>


