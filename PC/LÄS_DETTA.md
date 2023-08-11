För att kunna starta demot på denna dator så finns den körbara koden:

```bash
/home/luigi/Desktop/4/zybo-rt-sampler/PC2/demo.py
```

# VIKTIGT
I nuläget används ```/dev/video2``` för webkamera som kan ändras till något annat i:

```bash
/home/luigi/Desktop/4/zybo-rt-sampler/PC2/src/visual.py # i klassen Front
```

Och om man bygger med hjälp av Docker i `start.sh` så måste `start.sh` modifieras till att inkludera den `/dev/videoX` som man använder.

## Konfigurering av variabler
För att konfigurera variabler ska ***endast*** `/home/luigi/Desktop/4/zybo-rt-sampler/PC2/src/config.json` modifieras. Samtliga `config.py` eller `config.h` byggs utifrån `config.json`. När man gör ändringar i `config.json` så sparas ändringarna när man kör:

```bash
make clean
make config # Valfritt
make # Bygger applikation, bygger även config först
```

För att enbart köra med heatmaps:

```bash
python3 demo.py mimo
```

För att både köra med ljud och bild (dåligt ljud):

```bash
python3 demo.py miso
```

Om applikationen skulle hänga sig är en lösning att köra:

```bash
killall python3
```

FFT implementationen är otestad på denna dator, men körs vid `/home/luigi/Desktop/4/zybo-rt-sampler/PC2/application/manage.py`:

```bash
python3 manage.py runserver
```