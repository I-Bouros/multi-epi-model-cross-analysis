# multi-epi-model-cross-analysis
A collection of multiple epidemiological models used in the modelling of cases observed during the COVID-19 pandemic. Amongst the models curated
in this collection are:
- *PHE & Cambridge*: official model used by the UK government for policy making.
- *Roche*.

All features of our software are described in detail in our
[full API documentation](https://multi-epi-model-cross-analysis.readthedocs.io/en/latest/).

More details on branching process models and inference can be found in these
papers:

## References
[1] Birrell, Paul and Blake, Joshua and van Leeuwen, Edwin and , and Gent, Nick and De Angelis, Daniela (2020). [Real-time Nowcasting and Forecasting of COVID-19 Dynamics in England: the first wave?](https://www.medrxiv.org/content/early/2020/08/30/2020.08.24.20180737). In medRxiv. 

[2] Lemenuel-Diot, Annabelle and Clinch, Barry and Hurt, Aeron C. and Boutry, Paul and Laurent, Johann and Leddin, Mathias and Frings, Stefan and Charoin, Jean Eric (2020). [A COVID-19 transmission model informing medication development and supply chain needs](https://www.medrxiv.org/content/early/2020/12/02/2020.11.23.20237404). In medRxiv.

## Installation procedure
***
One way to install the module is to download the repositiory to your machine of choice and type the following commands in the terminal. 
```bash
git clone https://github.com/I-Bouros/multi-epi-model-cross-analysis.git
cd ../path/to/the/file
```

A different method to install this is using `pip`:

```bash
pip install -i https://test.pypi.org/simple/ epimodels==0.0.1
```

## Usage

```python
import epimodels

# create a contact matrix using mobility data e.g. from a POLYMOD matrix
epimodels.OneDimSampler(age_groups, polymod_matrix)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)