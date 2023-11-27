use std::fs::OpenOptions;
use std::io::{Write, self};

use rand::{seq::index::sample, thread_rng, Rng};

fn rand_cycle<R>(rng: &mut R) -> Vec<u8> 
where
    R: Rng
{
    sample(rng, 7, 3)
        .iter()
        .map(|i| i as u8)
        .collect()
}

fn oneline(cycle: &[u8]) -> Vec<u8> {
    let mut v: Vec<u8> = (0..7).collect();

    v[cycle[0] as usize] = cycle[1];
    v[cycle[1] as usize] = cycle[2];
    v[cycle[2] as usize] = cycle[0];

    v
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (ntest, ntrain, nvalidate) = match &args[1..] {
        [test, train, validate] => (
            test.parse().unwrap(),
            train.parse().unwrap(),
            validate.parse().unwrap(),
        ),
        _ => (200, 1000, 200)
    };

    gen(ntest, "test.txt")?;
    gen(ntrain, "train.txt")?;
    gen(nvalidate, "validate.txt")?;

    Ok(())
}


fn gen(n: usize, filename: &str) -> io::Result<()> {
    let mut rng = thread_rng();

    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(filename)?;

    for _ in 0..n {
        let cycle = rand_cycle(&mut rng);
        for i in &cycle {
            write!(f, "{} ", i)?;
        }
        for i in 0..7 {
            write!(f, "{} ", i)?;
        }
        write!(f, "\n")?;

        let permutation = oneline(&cycle[..]); 
        for i in &cycle {
            write!(f, "{} ", i)?;
        }
        for i in &permutation {
            write!(f, "{} ", i)?;
        }
        write!(f, "\n")?;
    }

    Ok(())
}
