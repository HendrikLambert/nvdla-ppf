#include "FilterBank.h"
#include "PPF.h"

#include <stdlib.h>
#include <string.h>
#include <iostream>

using namespace std;

enum class Mode : uint16_t
{
  NONE = 0,
  PRINT_WEIGHTS = 1 << 0,
  REVERSED_WEIGHTS = 1 << 1,
  TEST_FILTER = 1 << 2,
};

bool isFlagSet(Mode mode, Mode flag)
{
  return (static_cast<uint16_t>(mode) & static_cast<uint16_t>(flag)) == static_cast<uint16_t>(flag);
}

int main(int argc, char *argv[])
{
  unsigned channels = 4;
  unsigned taps = 256;
  unsigned nrSamplesPerIntegration = 64;
  Mode mode = Mode::NONE;

  WindowType window = KAISER;

  if (argc == 1)
  {
    ; // use default settings
  }
  else if (argc != 5)
  {
    cerr << "Usage: " << argv[0] << " [nrChannels] [nrTaps] [windowType] [options]" << endl;
    cerr << "       where windowType is one of HAMMING, BLACKMAN, GAUSSIAN, KAISER" << endl;
    return -1;
  }
  else
  {
    channels = atoi(argv[1]);
    taps = atoi(argv[2]);
    window = FilterBank::getWindowTypeFromString(argv[3]);
    mode = static_cast<Mode>(atoi(argv[4]));

    if (window == ERROR)
    {
      cerr << "WindowType should be one of HAMMING, BLACKMAN, GAUSSIAN, KAISER" << endl;
      return -1;
    }
  }

  FilterBank fb = FilterBank(false, channels, taps, window);

  if (isFlagSet(mode, Mode::REVERSED_WEIGHTS))
  {
    fb.reverseTaps();
  }

  if (isFlagSet(mode, Mode::PRINT_WEIGHTS))
  {
    fb.printWeights();
    // cout << "Printing weights" << endl;
  }

  if (isFlagSet(mode, Mode::TEST_FILTER))
  {
    // cout << "Testing filter with " << taps << " taps" << endl;
    // Test the filter
    FIR fir(taps, false);
    fir.setWeights(fb.getWeights(0));

    for (int i = 1; i < taps; i++)
    {
      fir.processNextSample(fcomplex(0, 0));
      // cout << sample.real() << endl;
    }

    fcomplex sample = fir.processNextSample(fcomplex(1, 0));
    cout << sample.real() << endl;

    for (int i = 1; i < taps; i++)
    {
      sample = fir.processNextSample(fcomplex(0, 0));
      cout << sample.real() << endl;
    }

    return 0;
  }

#if 0
  // Do some filtering
  // in = 1D array of size nrSamplesPerIntegration * nrChannels
  // out = [nrChannels][nrSamplesPerIntegration]

  vector<fcomplex> in(nrSamplesPerIntegration * channels);
  in[0] = fcomplex(1,0.0f);
/*
  for(int i=0; i< nrSamplesPerIntegration * channels; i++) {
    in[i] = fcomplex(i,0.0f);
    cout << "i = " << in[i] << endl;
  }
*/
  boost::multi_array<fcomplex, 2> out(boost::extents[channels][nrSamplesPerIntegration]);

  PPF ppf(channels, taps, nrSamplesPerIntegration, false);

  ppf.filter(in, out);
  for(int i=0; i< nrSamplesPerIntegration; i++) {
    cout << "out = " << out[0][i] << endl;
  }
#endif
#if 0
  ppf.getImpulseResponse(0, response);

  for(int i=0; i<taps; i++) {
    cout << response[i].real() << endl;
  }
#endif

#if 0
  vector<complex<float> > response(taps);
  PPF ppf(channels, taps);
  ppf.getFrequencyResponse(0, response);

  for(int i=0; i<taps; i+=2) {
    cout << response[i].real() << endl;
  }
#endif
}
