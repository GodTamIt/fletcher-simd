use byteorder::{ByteOrder, LittleEndian};
use fletcher_simd::{Fletcher128, Fletcher16, Fletcher32, Fletcher64};

#[test]
fn simple_fletcher16() {
    const DATA: &str = "abcdefgh";
    let mut fletcher = Fletcher16::new();

    fletcher.update_with_slice(DATA.as_bytes());

    assert_eq!(fletcher.value(), 0xF824);
}

#[test]
fn simple_fletcher32() {
    const DATA: &str = "abcdefgh";
    let mut fletcher = Fletcher32::new();

    fletcher.update_with_iter(
        DATA.as_bytes()
            .chunks(2)
            .map(|chunk| LittleEndian::read_u16(chunk)),
    );

    assert_eq!(fletcher.value(), 0xEBDE9590);
}

#[test]
fn simple_fletcher64() {
    const DATA: &str = "abcdefgh";
    let mut fletcher = Fletcher64::new();

    fletcher.update_with_iter(
        DATA.as_bytes()
            .chunks(4)
            .map(|chunk| LittleEndian::read_u32(chunk)),
    );

    assert_eq!(fletcher.value(), 0x312E2B27CCCAC8C6);
}

#[test]
fn simple_fletcher128() {
    const DATA: &str = "abcdefgh";
    let mut fletcher = Fletcher128::new();

    fletcher.update_with_iter(
        DATA.as_bytes()
            .chunks(8)
            .map(|chunk| LittleEndian::read_u64(chunk)),
    );

    assert_eq!(fletcher.value(), 0x68676665646362616867666564636261);
}

#[test]
fn lorem_fletcher16() {
    const DATA: &'static str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Interdum velit laoreet id donec ultrices tincidunt. Phasellus vestibulum lorem sed risus ultricies tristique nulla aliquet. Id cursus metus aliquam eleifend mi in. Condimentum vitae sapien pellentesque habitant morbi tristique. Fringilla est ullamcorper eget nulla facilisi etiam dignissim diam quis.";

    let mut fletcher = Fletcher16::new();

    fletcher.update_with_slice(DATA.as_bytes());

    assert_eq!(fletcher.value(), 0x51CF);
}

#[test]
fn lorem_fletcher32() {
    const DATA: &'static str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Interdum velit laoreet id donec ultrices tincidunt. Phasellus vestibulum lorem sed risus ultricies tristique nulla aliquet. Id cursus metus aliquam eleifend mi in. Condimentum vitae sapien pellentesque habitant morbi tristique. Fringilla est ullamcorper eget nulla facilisi etiam dignissim diam quis.";

    let mut fletcher = Fletcher32::new();

    fletcher.update_with_iter(
        DATA.as_bytes()
            .chunks(2)
            .map(|chunk| LittleEndian::read_u16(chunk)),
    );

    assert_eq!(fletcher.value(), 0xB1A48896);
}

#[test]
fn lorem_fletcher64() {
    const DATA: &'static str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Interdum velit laoreet id donec ultrices tincidunt. Phasellus vestibulum lorem sed risus ultricies tristique nulla aliquet. Id cursus metus aliquam eleifend mi in. Condimentum vitae sapien pellentesque habitant morbi tristique. Fringilla est ullamcorper eget nulla facilisi etiam dignissim diam quis.";

    let mut fletcher = Fletcher64::new();

    fletcher.update_with_iter(
        DATA.as_bytes()
            .chunks(4)
            .map(|chunk| LittleEndian::read_u32(chunk)),
    );

    assert_eq!(fletcher.value(), 0x72FFE298E896A028);
}

#[test]
fn lorem_fletcher128() {
    const DATA: &'static str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Interdum velit laoreet id donec ultrices tincidunt. Phasellus vestibulum lorem sed risus ultricies tristique nulla aliquet. Id cursus metus aliquam eleifend mi in. Condimentum vitae sapien pellentesque habitant morbi tristique. Fringilla est ullamcorper eget nulla facilisi etiam dignissim diam quis.";

    let mut fletcher = Fletcher128::new();

    fletcher.update_with_iter(
        DATA.as_bytes()
            .chunks(8)
            .map(|chunk| LittleEndian::read_u64(chunk)),
    );

    assert_eq!(fletcher.value(), 0xC6B64C7008FC4EC12C654FCFBC31506C);
}
