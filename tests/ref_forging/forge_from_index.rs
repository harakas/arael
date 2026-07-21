// A `Ref` must not be constructible from a bare number: it is only
// meaningful against the collection that issued it.
use arael::refs::{self, Ref};

struct Item;

fn main() {
    let mut items: refs::Vec<Item> = refs::Vec::new();
    items.push(Item);
    let forged: Ref<Item> = Ref::new(0);
    let _ = &items[forged];
}
