use std::{
    ops::{Index, IndexMut},
    slice,
    sync::Arc,
    usize,
};

#[derive(Debug)]
pub struct Tensor<T> {
    data: Arc<Box<[T]>>,
    shape: Vec<usize>,
    stride: Vec<usize>,
    offset: usize,
    length: usize,
}

impl<T: Copy + Clone + Default> Tensor<T> {
    fn get_stride(shape: &Vec<usize>) -> Vec<usize> {
        shape
            .iter()
            .rev()
            .scan(1, |acc, x| {
                let res = *acc;
                *acc *= x;
                Some(res)
            })
            .collect::<Vec<usize>>()
            .into_iter()
            .rev()
            .collect()
    }

    pub fn new(data: Vec<T>, shape: &Vec<usize>) -> Self {
        let length = data.len();
        Tensor {
            data: Arc::new(data.into_boxed_slice().try_into().unwrap()),
            shape: shape.clone(),
            stride: Self::get_stride(shape),
            offset: 0,
            length: length,
        }
    }

    pub fn default(shape: &Vec<usize>) -> Self {
        let length = shape.iter().product();
        let data = vec![T::default(); length];
        Self::new(data, shape)
    }

    pub fn data(&self) -> &[T] {
        &self.data[self.offset..][..self.length]
    }

    pub unsafe fn data_mut(&mut self) -> &mut [T] {
        let ptr = self.data.as_ptr().add(self.offset) as *mut T;
        slice::from_raw_parts_mut(ptr, self.length)
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn stride(&self) -> &Vec<usize> {
        &self.stride
    }

    pub fn size(&self) -> usize {
        self.length
    }

    // Reinterpret the tensor as a new shape while preserving total size.
    pub fn reshape(&mut self, new_shape: &Vec<usize>) -> &mut Self {
        let new_length: usize = new_shape.iter().product();
        if new_length != self.length {
            let old_shape = self.shape.clone();
            panic!("New shape {new_shape:?} does not match tensor of {old_shape:?}");
        }
        self.shape = new_shape.clone();
        self
    }

    pub fn slice(&self, start: usize, shape: &Vec<usize>) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(self.offset + start + new_length <= self.length);
        Tensor {
            data: self.data.clone(),
            shape: shape.clone(),
            stride: Self::get_stride(shape),
            offset: self.offset + start,
            length: new_length,
        }
    }
}

// Some helper functions for testing and debugging
impl Tensor<f32> {
    #[allow(unused)]
    pub fn close_to(&self, other: &Self, rel: f32) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let a = self.data();
        let b = other.data();

        return a.iter().zip(b).all(|(x, y)| float_eq(x, y, rel));
    }
    #[allow(unused)]
    pub fn print(&self) {
        println!(
            "shpae: {:?}, offset: {}, length: {}",
            self.shape, self.offset, self.length
        );
        let dim = self.shape()[self.shape().len() - 1];
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!("{:?}", &self.data()[start..][..dim]);
        }
    }
}

impl<T: Copy + Clone + Default> Tensor<T> {
    fn get_index(&self, indexs: &[usize]) -> usize {
        assert!(
            indexs.len() == self.shape.len(),
            "Index dimensions ({}) do not match tensor dimensions ({})",
            indexs.len(),
            self.shape.len()
        );
        assert!(
            indexs
                .iter()
                .enumerate()
                .all(|(i, &idx)| idx < self.shape[i]),
            "Index out of bounds input: {:?}, shape: {:?}, offset: {}",
            indexs,
            self.shape,
            self.offset
        );

        indexs
            .iter()
            .zip(self.stride.iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum::<usize>()
    }
}

impl<T: Copy + Clone + Default> Index<&[usize]> for Tensor<T> {
    type Output = T;

    fn index(&self, indexs: &[usize]) -> &Self::Output {
        let index = self.get_index(indexs);
        &self.data()[index]
    }
}

impl<T: Copy + Clone + Default> IndexMut<&[usize]> for Tensor<T> {
    fn index_mut(&mut self, indexs: &[usize]) -> &mut Self::Output {
        let index = self.get_index(indexs);
        unsafe { &mut self.data_mut()[index] }
    }
}

#[inline]
pub fn float_eq(x: &f32, y: &f32, rel: f32) -> bool {
    (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.0
}
