import binary_fitter, pandas, os
    
if __name__=='__main__':
    lib = pandas.read_excel('VLE_data_DOE.xlsx')

    # Uncomment this line and modify path to change the root directory of your python installation
    # os.environ['RPPREFIX'] = r'D:\Code\REFPROP-cmake\build\10\Release\\'

    # For serial evaluation, you can just call the deap_optimizer function
    aliases = {'DECANE': os.path.abspath('DECANE.FLD'), 'PROPANE': os.path.abspath('PROPANE.FLD')}
    binary_fitter.deap_optimizer(binary_fitter.collect_binary(lib, ('DECANE', 'PROPANE')), aliases=aliases)