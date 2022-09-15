use clap::ArgEnum;
use itertools::Itertools;
use log::{error, info};
use std::error::Error;
use std::fs;
use std::io;
use std::path::Path;

mod pretty;
mod tree_sitter;

#[derive(ArgEnum, Clone, Copy, Debug)]
pub enum Language {
    Json,
    Ocaml,
    Rust,
}

/// A Node from tree-sitter is turned into into a list of atoms
#[derive(Clone, Debug, PartialEq)]
pub enum Atom {
    Blankline,
    Empty,
    Hardline,
    IndentStart,
    IndentEnd,
    Leaf { content: String, id: usize },
    Literal(String),
    Softline { spaced: bool },
    Space,
}

pub fn formatter(
    input: &mut dyn io::Read,
    output: &mut dyn io::Write,
    language: Language,
    check_idempotence: bool,
) -> Result<(), Box<dyn Error>> {
    let content = read_input(input)?;
    let query = read_query(language)?;

    // All the work related to tree-sitter and the query is done here
    let query_result = tree_sitter::apply_query(&content, &query, language)?;
    let mut atoms = query_result.atoms;

    // Various post-processing of whitespace
    //
    // TODO: Make sure these aren't unnecessarily inefficient, in terms of
    // recreating a vector of atoms over and over.
    log::debug!("Before post-processing: {atoms:?}");
    put_before(&mut atoms, Atom::IndentEnd, Atom::Space);
    let mut atoms = trim_following(&atoms, Atom::Blankline, Atom::Space);
    put_before(&mut atoms, Atom::Hardline, Atom::Blankline);
    put_before(&mut atoms, Atom::IndentEnd, Atom::Hardline);
    let atoms = trim_following(&atoms, Atom::Hardline, Atom::Space);
    let atoms = clean_up_consecutive(&atoms, Atom::Space);
    let mut atoms = clean_up_consecutive(&atoms, Atom::Hardline);
    ensure_final_hardline(&mut atoms);
    log::debug!("Final list of atoms: {atoms:?}");

    // Pretty-print atoms
    let rendered = pretty::render(&atoms, query_result.indent_level)?;
    let trimmed = trim_trailing_spaces(&rendered);

    if check_idempotence {
        idempotence_check(&trimmed, language)?
    }

    write!(output, "{trimmed}")?;

    Ok(())
}

fn read_input(input: &mut dyn io::Read) -> Result<String, Box<dyn Error>> {
    let mut content = String::new();
    input.read_to_string(&mut content)?;
    Ok(content)
}

fn read_query(language: Language) -> Result<String, Box<dyn Error>> {
    Ok(fs::read_to_string(Path::new(&str::to_lowercase(
        format!("languages/queries/{:?}.scm", language).as_str(),
    )))?)
}

fn clean_up_consecutive(atoms: &Vec<Atom>, atom: Atom) -> Vec<Atom> {
    let filtered = atoms.split(|a| *a == atom).filter(|chain| chain.len() > 0);

    Itertools::intersperse(filtered, &[atom.clone()])
        .flatten()
        .map(|a| a.clone())
        .collect_vec()
}

fn trim_following(atoms: &Vec<Atom>, delimiter: Atom, skip: Atom) -> Vec<Atom> {
    let trimmed = atoms.split(|a| *a == delimiter).map(|slice| {
        slice
            .into_iter()
            .skip_while(|a| **a == skip)
            .collect::<Vec<_>>()
    });

    Itertools::intersperse(trimmed, vec![&delimiter])
        .flatten()
        .map(|a| a.clone())
        .collect_vec()
}

fn put_before(atoms: &mut Vec<Atom>, before: Atom, after: Atom) {
    for i in 0..atoms.len() - 1 {
        if atoms[i] == after && atoms[i + 1] == before {
            for j in i + 1..atoms.len() {
                if atoms[j] != before && atoms[j] != after {
                    // stop looking
                    break;
                }
                if atoms[j] == before {
                    // switch
                    atoms[i] = before.clone();
                    atoms[j] = after.clone();
                    break;
                }
            }
        }
    }
}

fn clean_space_between_indent_ends(atoms: &mut Vec<Atom>) {
    for i in 0..atoms.len() - 2 {
        if atoms[i] == Atom::IndentEnd
            && atoms[i + 1] == Atom::Space
            && atoms[i + 2] == Atom::IndentEnd
        {
            atoms[i + 1] = Atom::Empty;
        }
    }
}

fn ensure_final_hardline(atoms: &mut Vec<Atom>) {
    if let Some(Atom::Hardline) = atoms.last() {
    } else {
        atoms.push(Atom::Hardline);
    }
}

fn trim_trailing_spaces(s: &str) -> String {
    Itertools::intersperse(s.split('\n').map(|line| line.trim_end()), "\n").collect::<String>()
}

fn idempotence_check(content: &str, language: Language) -> Result<(), Box<dyn Error>> {
    info!("Checking for idempotence ...");

    let mut input = content.as_bytes();
    let mut output = io::BufWriter::new(Vec::new());
    formatter(&mut input, &mut output, language, false)?;
    let reformatted = String::from_utf8(output.into_inner()?)?;

    if content == reformatted {
        Ok(())
    } else {
        error!(
            "Failed idempotence check. First output: {content} - Reformatted output: {reformatted}"
        );
        Err("The formatter is not idempotent on this input. Please log an error.".into())
    }
}
