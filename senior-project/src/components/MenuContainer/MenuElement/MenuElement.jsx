/**
 * MenuElement Component: Button in MenuContainer
 */

/**
 * MenuElement()
 * 
 * NAME
 *    MenuElement() - Handles single elements in the Menu
 * 
 * SYNOPSIS
 *    React.JSX.Element MenuElement(a_props)
 *      a_props --> properties
 *          a_props.className --> Class name for css
 *          a_props.onClick   --> Function to be executed when element is clicked
 *          a_props.children  --> Text to be shown in MenuElement
 * 
 * DESCRIPTION
 *    This function creates an element in the menu
 * 
 * RETURNS 
 *    Returns jsx element for an element in the menu
 */
export default function MenuElement(a_props) {
    return (
        <div className={a_props.className} onClick={a_props.onClick}>
            {a_props.children}
        </div>
    );
}