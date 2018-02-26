import configparser

def process_config(conf_file):
  """process configure file to generate CommonParams, DataSetParams, NetParams 

  Args:
    conf_file: configure file path 
  Returns:
    CommonParams
    dataset_params
  """
  common_params = {}
  dataset_params = {}
  models_params = {}

  #configure_parser
  config = configparser.ConfigParser()
  config.read(conf_file)

  #sections and options
  for section in config.sections():
    #construct common_params
    if section == 'Common':
      for option in config.options(section):
        common_params[option] = config.get(section, option)
    #construct dataset
    if section == 'DataSet':
      for option in config.options(section):
        dataset_params[option] = config.get(section, option)
    # construct dataset
    if section == 'Models':
      for option in config.options(section):
        models_params[option] = config.get(section, option)

  return common_params, dataset_params, models_params