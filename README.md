# Signal Resolution
This program applied some processes respectively to resolve the given composited signals. In this way, generated voice can be resolved.
These processes are below respectively:
1. The **composit signal** ,which is input, is converted to a graphic in the **frequence domain**.
In this way, the peak signal values can be distinguished.wi
2. After that, **band pass filter** is applied for the peak values.
3. 2 **demodulation methods** which are **cosine demodulation** and **absolute demodulation** are applied separately to band pass filtered signals.
4. After the demodulation application, **low pass filter** is also applied separately to get the results.
5. Finally, the **results are converted to the wav file formats**. When they are running the voice message will be released.
