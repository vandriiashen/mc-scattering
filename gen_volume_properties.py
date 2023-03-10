import numpy as np
import itertools

def gen_uniform(mode = 'test', beta_a = 1.1, beta_b = 1.0):
    '''Generates a list of volume parameters.
    The goal is to have a variety of different cylinders with different cavities inside.
    One of the details to note is that cavity detection might depend on the cylinder thickness in the cavity location.
    Beta distribution with these default parameters was used to make the distribution of these thicknesses in a set more uniform.
    
    :param mode: Mode of generation: training, validation, test
    :type mode: :class:`str`
    :param beta_a: Parameter of beta distribution
    :type beta_a: :class:`float`
    :param beta_b: Parameter of beta distribution
    :type beta_b: :class:`float`
    '''
    if mode == 'train':
        seed = 0
        num_per_r = 100
    elif mode == 'val':
        seed = 1
        num_per_r = 25
    elif mode == 'test':
        seed = 2
        num_per_r = 100
    np.random.seed(seed)
    cavity_r = np.linspace(-0.99, 0.99, num_per_r, endpoint = True)
    np.random.shuffle(cavity_r)
    cavity_sizes = np.arange(0.1, 1.1, 0.1)
    
    comb = itertools.product(cavity_r, cavity_sizes)
    comb = list(comb)
    data_spec = np.zeros((len(comb), 9))
    data_spec[:,0] = np.arange(0, len(comb))
    
    for i in range(len(comb)):
        data_spec[i,1] = comb[i][1]
        data_spec[i,2] = 1+25.*np.random.beta(beta_a, beta_b, size=(1))
        data_spec[i,3] = np.random.uniform(20., 55., size=(1,))
        data_spec[i,4] = comb[i][0] * data_spec[i,2]
        data_spec[i,5] = np.random.uniform(-0.8, 0.8, size=(1,)) * 0.5 * data_spec[i,3]
        data_spec[i,6] = np.random.uniform(0.7, 1.3, size=(1,))
        data_spec[i,7] = np.random.uniform(0.7, 1.3, size=(1,))
        data_spec[i,8] = np.random.uniform(0.7, 1.3, size=(1,))
    
    return data_spec

def gen_difficult_test(beta_a = 1.1, beta_b = 1.0):
    '''Generates a list of volume parameters.
    In this case the defect is on the axis of rotation.
    
    :param mode: Mode of generation: training, validation, test
    :type mode: :class:`str`
    :param beta_a: Parameter of beta distribution
    :type beta_a: :class:`float`
    :param beta_b: Parameter of beta distribution
    :type beta_b: :class:`float`
    '''
    seed = 3
    num_per_r = 100
    np.random.seed(seed)
    cavity_r = np.repeat(0., num_per_r)
    cavity_sizes = np.arange(0.1, 1.1, 0.1)
    
    comb = itertools.product(cavity_r, cavity_sizes)
    comb = list(comb)
    data_spec = np.zeros((len(comb), 9))
    data_spec[:,0] = np.arange(0, len(comb))
    
    for i in range(len(comb)):
        data_spec[i,1] = comb[i][1]
        data_spec[i,2] = 1+25.*np.random.beta(beta_a, beta_b, size=(1))
        data_spec[i,3] = np.random.uniform(20., 55., size=(1,))
        data_spec[i,4] = comb[i][0] * data_spec[i,2]
        data_spec[i,5] = np.random.uniform(-0.8, 0.8, size=(1,)) * 0.5 * data_spec[i,3]
        data_spec[i,6] = np.random.uniform(0.7, 1.3, size=(1,))
        data_spec[i,7] = np.random.uniform(0.7, 1.3, size=(1,))
        data_spec[i,8] = np.random.uniform(0.7, 1.3, size=(1,))
    
    return data_spec

if __name__ == '__main__':
    data_spec_train = gen_uniform(mode = 'train', beta_a = 1.1, beta_b = 1.0)
    np.savetxt('data/data_spec_train.csv', data_spec_train, delimiter=',', header='proj_num,size,cyl_r,cyl_h,cav_r,cav_z,el_x,el_y,el_z')
    data_spec_val = gen_uniform(mode = 'val', beta_a = 1.1, beta_b = 1.0)
    np.savetxt('data/data_spec_val.csv', data_spec_val, delimiter=',', header='proj_num,size,cyl_r,cyl_h,cav_r,cav_z,el_x,el_y,el_z')
    data_spec_test = gen_uniform(mode = 'test', beta_a = 1.1, beta_b = 1.0)
    np.savetxt('data/data_spec_test.csv', data_spec_test, delimiter=',', header='proj_num,size,cyl_r,cyl_h,cav_r,cav_z,el_x,el_y,el_z')
    data_spec_test2 = gen_difficult_test(beta_a = 1.1, beta_b = 1.0)
    np.savetxt('data/data_spec_test2.csv', data_spec_test2, delimiter=',', header='proj_num,size,cyl_r,cyl_h,cav_r,cav_z,el_x,el_y,el_z')
