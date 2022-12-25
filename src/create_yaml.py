import sys
pkg_path = '..'
sys.path.append(pkg_path)
import yaml
import os




def create_yaml(start_time=50, car_lw=[6, 4]):
    yaml_dict = {
        'start_time': start_time,
        'car_lw': car_lw
    }
    yaml_config = 't'+str(start_time)+'_lw'+str(car_lw[0])+str(car_lw[1])
    yaml_filename = yaml_config+'.yaml'
    with open(os.path.join('yaml_configurations', yaml_filename), 'w') as file:
        yaml.dump(yaml_dict, file)
        print ("Successfully created %s" % yaml_filename)
    return yaml_config

def get_config_and_result_folder(yaml_config, create=False):
    yaml_filename = yaml_config+'.yaml'
    with open(os.path.join('yaml_configurations', yaml_filename), 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader) 
        print ("Successfully loaded %s" % yaml_filename)
    result_folder_name = os.path.join('results_paper', yaml_config)
    if create:
        try:
            os.mkdir(result_folder_name)
        except OSError:
            print ("Creation of the directory %s failed" % result_folder_name)
        else:
            print ("Successfully created the directory %s " % result_folder_name)
    return config, result_folder_name

def main():

    for start_time in [0, 50]:
        for car_lw in [[6, 4], [4, 2]]:
            yaml_config = create_yaml(start_time, car_lw)
            config, result_folder_name = get_config_and_result_folder(yaml_config, create=True)
            print(config)
            print(result_folder_name)
            
            
    
    # for num_lstms in [1, 3, 8]:
    #     for bidirectional in [True, False]:
    #         for end_mask in [True, False]:
    #             for num_epochs in [200]:# [50]:
    #                 yaml_filename = create_yaml(num_lstms, bidirectional, end_mask, num_epochs)
    #                 with open(os.path.join(pkg_path, 'notebooks_new/yaml_config', yaml_filename), 'r') as file:
    #                     args = yaml.load(file, Loader=yaml.FullLoader)
    #                     print('yaml file load succeeded.')

if __name__ == '__main__':
    main()